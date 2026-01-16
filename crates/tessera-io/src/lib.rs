//! # tessera-io
//!
//! RFC-4180-style CSV reader with schema inference, and a CSV writer.
//!
//! * Field separators/quotes are located with `memchr`.
//! * Inference per column: `Int64` if every non-null field parses as i64, else
//!   `Float64`, else `Boolean` (`true`/`false`), else `Utf8`. By default every
//!   row is inspected; [`CsvOptions::infer_rows`] limits this to the first N rows
//!   (a later field that does not fit the inferred type is then an error).
//! * An **unquoted empty** field is null; a quoted empty field (`""`) is the
//!   empty string. The writer uses the same convention, so data round-trips.
//! * The body is split at newline boundaries into chunks that are processed in
//!   parallel with rayon when it contains no `"` (a quoted field may contain a
//!   newline, so inputs with quotes use a single chunk).
//! * Per chunk: the tokenizer records `(offset, length)` spans for the wanted
//!   columns (no owned `String` per field); each (chunk, column) pair is
//!   inferred independently and the per-chunk results are merged; then every
//!   (chunk, column) pair is parsed straight into a typed builder and the
//!   chunk arrays are concatenated. Parallelism is therefore bounded by the
//!   chunk count, not the column count.
//! * [`CsvOptions::projection`] keeps only the named columns: other fields are
//!   skipped by the tokenizer and never parsed.
//! * A single chunk (or the whole input when it has quotes) must be smaller than
//!   4 GiB ([`CsvError::TooLarge`]).
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

use std::borrow::Cow;
use std::fmt::Write as _;
use std::path::Path;

use rayon::prelude::*;
use tessera_arrow::{
    Array, ArrowError, Bitmap, BooleanBuilder, Buffer, PrimitiveArray, PrimitiveBuilder, Utf8Array,
    Utf8Builder,
};
use tessera_core::{CoreError, DataFrame, DataType, Scalar, Series};
use thiserror::Error;

/// CSV errors.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CsvError {
    /// Input is not UTF-8.
    #[error("input is not valid utf-8")]
    Utf8,
    /// A quoted field was not closed.
    #[error("unterminated quoted field starting on record {0}")]
    UnterminatedQuote(usize),
    /// Garbage after a closing quote.
    #[error("unexpected character after closing quote on record {0}")]
    BadQuote(usize),
    /// A record has the wrong number of fields.
    #[error("record {record} has {got} fields, expected {expected}")]
    FieldCount {
        /// 0-based record index (header = 0).
        record: usize,
        /// Fields found.
        got: usize,
        /// Fields expected.
        expected: usize,
    },
    /// No header line.
    #[error("empty input: missing header")]
    Empty,
    /// A field outside the inference window does not fit the inferred type.
    #[error("record {record}: field in column {column:?} does not parse as {expected:?}")]
    ParseField {
        /// 0-based record index (header = 0).
        record: usize,
        /// Column name.
        column: String,
        /// The inferred type.
        expected: DataType,
    },
    /// A projected column is not in the header.
    #[error("projected column not found: {0:?}")]
    UnknownColumn(String),
    /// A chunk (or a whole input with quotes) is 4 GiB or larger.
    #[error("a CSV chunk must be smaller than 4 GiB")]
    TooLarge,
    /// Reading the file failed.
    #[error("io error: {0}")]
    Io(String),
    /// DataFrame construction failed.
    #[error(transparent)]
    Core(#[from] CoreError),
}

impl From<ArrowError> for CsvError {
    fn from(e: ArrowError) -> Self {
        CsvError::Core(CoreError::from(e))
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, CsvError>;

/// Options for [`read_csv`] / [`read_csv_bytes_with`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CsvOptions {
    /// Infer column types from only the first N data rows (`None` = all rows).
    pub infer_rows: Option<usize>,
    /// Keep only these columns, in this order (`None` = all, header order).
    pub projection: Option<Vec<String>>,
    /// Chunk count (`None` = automatic; 1 = sequential).
    pub chunks: Option<usize>,
}

impl CsvOptions {
    /// Default options (infer over all rows, all columns, automatic chunking).
    pub fn new() -> Self {
        Self::default()
    }
    /// Infer types from the first `n` data rows only.
    pub fn with_infer_rows(mut self, n: usize) -> Self {
        self.infer_rows = Some(n);
        self
    }
    /// Parse only the named columns, returned in this order.
    pub fn with_projection<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.projection = Some(cols.into_iter().map(Into::into).collect());
        self
    }
    /// Use an explicit chunk count (1 = sequential).
    pub fn with_chunks(mut self, n: usize) -> Self {
        self.chunks = Some(n);
        self
    }
}

const QUOTED: u32 = 1 << 30;
const ESCAPED: u32 = 1 << 31;
const LEN_MASK: u32 = QUOTED - 1;

/// Location of one field's text inside a chunk (8 bytes). For quoted fields the
/// span covers the text between the quotes (`""` escapes still doubled).
#[derive(Debug, Clone, Copy)]
struct Span {
    start: u32,
    meta: u32,
}

impl Span {
    fn new(start: usize, len: usize, quoted: bool, escaped: bool) -> Result<Span> {
        let start = u32::try_from(start).map_err(|_| CsvError::TooLarge)?;
        let len = u32::try_from(len).map_err(|_| CsvError::TooLarge)?;
        if len > LEN_MASK {
            return Err(CsvError::TooLarge);
        }
        let mut meta = len;
        if quoted {
            meta |= QUOTED;
        }
        if escaped {
            meta |= ESCAPED;
        }
        Ok(Span { start, meta })
    }
    fn len(self) -> usize {
        (self.meta & LEN_MASK) as usize
    }
    fn is_null(self) -> bool {
        self.meta == 0
    }
    fn text(self, chunk: &str) -> Cow<'_, str> {
        let a = self.start as usize;
        let t = chunk.get(a..a + self.len()).unwrap_or("");
        if self.meta & ESCAPED != 0 {
            Cow::Owned(t.replace("\"\"", "\""))
        } else {
            Cow::Borrowed(t)
        }
    }
}

/// Scan one record starting at `pos`, calling `sink(field_index, span)` per
/// field. Returns the position after the line terminator and the field count.
fn scan_record(
    b: &[u8],
    mut pos: usize,
    rec: usize,
    mut sink: impl FnMut(usize, Span),
) -> Result<(usize, usize)> {
    let mut n = 0;
    loop {
        if pos < b.len() && b[pos] == b'"' {
            let start = pos + 1;
            let mut i = start;
            let mut escaped = false;
            let close = loop {
                let q = memchr::memchr(b'"', &b[i..]).ok_or(CsvError::UnterminatedQuote(rec))?;
                let close = i + q;
                i = close + 1;
                if b.get(i) == Some(&b'"') {
                    escaped = true;
                    i += 1;
                } else {
                    break close;
                }
            };
            sink(n, Span::new(start, close - start, true, escaped)?);
            n += 1;
            pos = i;
            match b.get(pos) {
                Some(b',') => pos += 1,
                Some(b'\n') | Some(b'\r') | None => break,
                Some(_) => return Err(CsvError::BadQuote(rec)),
            }
        } else {
            let end = memchr::memchr3(b',', b'\n', b'\r', &b[pos..]).map_or(b.len(), |e| pos + e);
            sink(n, Span::new(pos, end - pos, false, false)?);
            n += 1;
            pos = end;
            if b.get(pos) == Some(&b',') {
                pos += 1;
            } else {
                break;
            }
        }
    }
    // Consume line terminator (\n, \r\n or \r).
    if b.get(pos) == Some(&b'\r') {
        pos += 1;
    }
    if b.get(pos) == Some(&b'\n') {
        pos += 1;
    }
    Ok((pos, n))
}

/// Spans of the wanted columns for one chunk, column-major.
struct ChunkSpans {
    rows: usize,
    cols: Vec<Vec<Span>>,
}

/// Tokenize one chunk. `field_to_out[f]` is the output column of header field
/// `f` (or `None` when projected away). Record numbers are chunk-local.
fn tokenize_chunk(
    chunk: &str,
    n_fields: usize,
    field_to_out: &[Option<usize>],
    n_out: usize,
) -> Result<ChunkSpans> {
    let b = chunk.as_bytes();
    let est = memchr::memchr_iter(b'\n', b).count() + 1;
    let mut cols: Vec<Vec<Span>> = (0..n_out).map(|_| Vec::with_capacity(est)).collect();
    let mut pos = 0;
    let mut rows = 0;
    while pos < b.len() {
        let (next, n) = scan_record(b, pos, rows, |f, sp| {
            if let Some(Some(o)) = field_to_out.get(f) {
                cols[*o].push(sp);
            }
        })?;
        if n != n_fields {
            return Err(CsvError::FieldCount {
                record: rows,
                got: n,
                expected: n_fields,
            });
        }
        pos = next;
        rows += 1;
    }
    Ok(ChunkSpans { rows, cols })
}

/// Record number carried by an error, if any.
fn record_of(e: &CsvError) -> Option<usize> {
    match e {
        CsvError::UnterminatedQuote(r) | CsvError::BadQuote(r) => Some(*r),
        CsvError::FieldCount { record, .. } | CsvError::ParseField { record, .. } => Some(*record),
        _ => None,
    }
}

/// Turn a chunk-local record number into a global one (`by` = body rows before
/// the chunk + 1 for the header).
fn shift(e: CsvError, by: usize) -> CsvError {
    match e {
        CsvError::UnterminatedQuote(r) => CsvError::UnterminatedQuote(r + by),
        CsvError::BadQuote(r) => CsvError::BadQuote(r + by),
        CsvError::FieldCount {
            record,
            got,
            expected,
        } => CsvError::FieldCount {
            record: record + by,
            got,
            expected,
        },
        CsvError::ParseField {
            record,
            column,
            expected,
        } => CsvError::ParseField {
            record: record + by,
            column,
            expected,
        },
        other => other,
    }
}

/// Which types every non-null value seen so far is compatible with.
#[derive(Clone, Copy)]
struct Flags {
    int: bool,
    float: bool,
    boolean: bool,
    any: bool,
}

impl Flags {
    const START: Flags = Flags {
        int: true,
        float: true,
        boolean: true,
        any: false,
    };
    fn merge(self, o: Flags) -> Flags {
        Flags {
            int: self.int && o.int,
            float: self.float && o.float,
            boolean: self.boolean && o.boolean,
            any: self.any || o.any,
        }
    }
    fn data_type(self) -> DataType {
        if !self.any {
            DataType::Utf8
        } else if self.int {
            DataType::Int64
        } else if self.float {
            DataType::Float64
        } else if self.boolean {
            DataType::Boolean
        } else {
            DataType::Utf8
        }
    }
}

/// Infer the compatible types of one chunk's column.
fn infer_chunk(chunk: &str, spans: &[Span]) -> Flags {
    let mut f = Flags::START;
    for sp in spans {
        if sp.is_null() {
            continue;
        }
        f.any = true;
        let t = sp.text(chunk);
        let v: &str = &t;
        let mut int_ok = false;
        if f.int {
            int_ok = v.parse::<i64>().is_ok();
            f.int = int_ok;
        }
        if f.float && !int_ok {
            f.float = v.parse::<f64>().is_ok();
        }
        if f.boolean {
            f.boolean = v == "true" || v == "false";
        }
        if !(f.int || f.float || f.boolean) {
            break;
        }
    }
    f
}

/// Parse one chunk's column into a typed array of type `dt`. Errors carry a
/// chunk-local record number.
fn build_chunk(chunk: &str, spans: &[Span], dt: DataType, name: &str) -> Result<Array> {
    let bad = |i: usize| CsvError::ParseField {
        record: i,
        column: name.to_owned(),
        expected: dt,
    };
    Ok(match dt {
        DataType::Int64 | DataType::Int32 => {
            let mut b = PrimitiveBuilder::<i64>::with_capacity(spans.len());
            for (i, sp) in spans.iter().enumerate() {
                if sp.is_null() {
                    b.push(None);
                } else {
                    b.push(Some(sp.text(chunk).parse().map_err(|_| bad(i))?));
                }
            }
            Array::Int64(b.finish())
        }
        DataType::Float64 => {
            let mut b = PrimitiveBuilder::<f64>::with_capacity(spans.len());
            for (i, sp) in spans.iter().enumerate() {
                if sp.is_null() {
                    b.push(None);
                } else {
                    b.push(Some(sp.text(chunk).parse().map_err(|_| bad(i))?));
                }
            }
            Array::Float64(b.finish())
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::new();
            for (i, sp) in spans.iter().enumerate() {
                if sp.is_null() {
                    b.push(None);
                } else {
                    match &*sp.text(chunk) {
                        "true" => b.push(Some(true)),
                        "false" => b.push(Some(false)),
                        _ => return Err(bad(i)),
                    }
                }
            }
            Array::Boolean(b.finish())
        }
        DataType::Utf8 => {
            let mut b = Utf8Builder::new();
            for sp in spans {
                b.push((!sp.is_null()).then(|| sp.text(chunk)).as_deref());
            }
            Array::Utf8(b.finish()?)
        }
    })
}

/// Validity bitmap of the concatenation, or `None` when no part has nulls.
fn concat_validity(parts: &[(usize, Option<&Bitmap>)]) -> Option<Bitmap> {
    if parts
        .iter()
        .all(|(_, v)| v.is_none_or(|v| v.unset_bits() == 0))
    {
        return None;
    }
    Some(Bitmap::from_bools(parts.iter().flat_map(|(len, v)| {
        (0..*len).map(move |i| v.is_none_or(|v| v.get(i)))
    })))
}

fn concat_primitive<T: tessera_arrow::NativeType>(
    parts: &[&PrimitiveArray<T>],
) -> Result<PrimitiveArray<T>> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut values = Vec::with_capacity(total);
    for p in parts {
        values.extend_from_slice(p.values().as_slice());
    }
    let meta: Vec<_> = parts.iter().map(|p| (p.len(), p.validity())).collect();
    Ok(PrimitiveArray::try_new(
        Buffer::from(values),
        concat_validity(&meta),
    )?)
}

/// Concatenate same-typed arrays (all freshly built, so offset 0).
fn concat(mut parts: Vec<Array>, dt: DataType) -> Result<Array> {
    if parts.len() == 1 {
        if let Some(a) = parts.pop() {
            return Ok(a);
        }
    }
    Ok(match dt {
        DataType::Int64 | DataType::Int32 => {
            let v: Vec<_> = parts
                .iter()
                .filter_map(|a| {
                    if let Array::Int64(x) = a {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();
            Array::Int64(concat_primitive(&v)?)
        }
        DataType::Float64 => {
            let v: Vec<_> = parts
                .iter()
                .filter_map(|a| {
                    if let Array::Float64(x) = a {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();
            Array::Float64(concat_primitive(&v)?)
        }
        DataType::Boolean => {
            let mut b = BooleanBuilder::new();
            for a in &parts {
                if let Array::Boolean(x) = a {
                    for v in x.iter() {
                        b.push(v);
                    }
                }
            }
            Array::Boolean(b.finish())
        }
        DataType::Utf8 => {
            let v: Vec<&Utf8Array> = parts
                .iter()
                .filter_map(|a| {
                    if let Array::Utf8(x) = a {
                        Some(x)
                    } else {
                        None
                    }
                })
                .collect();
            let rows: usize = v.iter().map(|p| p.len()).sum();
            let bytes: usize = v.iter().map(|p| p.data().len()).sum();
            let mut data = Vec::with_capacity(bytes);
            let mut offsets: Vec<i32> = Vec::with_capacity(rows + 1);
            offsets.push(0);
            for p in &v {
                let base = i32::try_from(data.len()).map_err(|_| ArrowError::OffsetOverflow)?;
                data.extend_from_slice(p.data().as_slice());
                for o in p.offsets().as_slice().iter().skip(1) {
                    offsets.push(base.checked_add(*o).ok_or(ArrowError::OffsetOverflow)?);
                }
            }
            let meta: Vec<_> = v.iter().map(|p| (p.len(), p.validity())).collect();
            Array::Utf8(Utf8Array::try_new(
                Buffer::from(offsets),
                Buffer::from(data),
                concat_validity(&meta),
            )?)
        }
    })
}

/// Split `body` into at most `n` pieces ending right after a `\n`.
fn chunk_bounds(b: &[u8], n: usize) -> Vec<usize> {
    let mut bounds = vec![0usize];
    if n <= 1 {
        bounds.push(b.len());
        return bounds;
    }
    let target = b.len().div_ceil(n).max(1);
    let mut start = 0;
    while start < b.len() {
        let want = (start + target).min(b.len());
        let from = want.saturating_sub(1);
        let end = match memchr::memchr(b'\n', &b[from..]) {
            Some(p) => from + p + 1,
            None => b.len(),
        };
        let end = end.max(start + 1).min(b.len());
        bounds.push(end);
        start = end;
    }
    if bounds.len() == 1 {
        bounds.push(b.len());
    }
    bounds
}

/// Parse CSV bytes (with header row) into a [`DataFrame`], processing chunks in
/// parallel for inputs over 64 KiB without quotes.
pub fn read_csv_bytes(bytes: &[u8]) -> Result<DataFrame> {
    read_csv_bytes_with(bytes, &CsvOptions::default())
}

/// Like [`read_csv_bytes`] with an explicit chunk count (1 = sequential).
pub fn read_csv_bytes_chunked(bytes: &[u8], n_chunks: usize) -> Result<DataFrame> {
    read_csv_bytes_with(bytes, &CsvOptions::default().with_chunks(n_chunks))
}

/// Read a CSV file (the whole file is loaded into memory first).
pub fn read_csv(path: impl AsRef<Path>, opts: &CsvOptions) -> Result<DataFrame> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| CsvError::Io(e.to_string()))?;
    read_csv_bytes_with(&bytes, opts)
}

/// Read only the header line of a CSV file and return its column names.
/// The header is taken to end at the first newline (a header field containing a
/// quoted newline is not supported).
pub fn read_csv_header(path: impl AsRef<Path>) -> Result<Vec<String>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path.as_ref()).map_err(|e| CsvError::Io(e.to_string()))?;
    let mut line = Vec::new();
    std::io::BufReader::new(file)
        .read_until(b'\n', &mut line)
        .map_err(|e| CsvError::Io(e.to_string()))?;
    let s = std::str::from_utf8(&line).map_err(|_| CsvError::Utf8)?;
    if s.is_empty() {
        return Err(CsvError::Empty);
    }
    let mut header: Vec<String> = Vec::new();
    scan_record(s.as_bytes(), 0, 0, |_, sp| {
        header.push(sp.text(s).into_owned())
    })?;
    Ok(header)
}

/// Parse CSV bytes with [`CsvOptions`].
pub fn read_csv_bytes_with(bytes: &[u8], opts: &CsvOptions) -> Result<DataFrame> {
    let s = std::str::from_utf8(bytes).map_err(|_| CsvError::Utf8)?;
    let b = s.as_bytes();
    if b.is_empty() {
        return Err(CsvError::Empty);
    }
    // Header (record 0).
    let mut header: Vec<String> = Vec::new();
    let (body_start, _) = scan_record(b, 0, 0, |_, sp| header.push(sp.text(s).into_owned()))?;
    let n_fields = header.len();
    // Output columns: (header index, name).
    let outputs: Vec<usize> = match &opts.projection {
        None => (0..n_fields).collect(),
        Some(names) => names
            .iter()
            .map(|n| {
                header
                    .iter()
                    .position(|h| h == n)
                    .ok_or_else(|| CsvError::UnknownColumn(n.clone()))
            })
            .collect::<Result<_>>()?,
    };
    let mut field_to_out = vec![None; n_fields];
    for (o, &h) in outputs.iter().enumerate() {
        if field_to_out[h].is_some() {
            return Err(CoreError::Duplicate(header[h].clone()).into());
        }
        field_to_out[h] = Some(o);
    }
    let n_out = outputs.len();

    // Chunk the body.
    let body = s.get(body_start..).unwrap_or("");
    let n_chunks = match opts.chunks {
        Some(n) => n,
        None if body.len() > 64 * 1024 => rayon::current_num_threads() * 4,
        None => 1,
    };
    let n_chunks = if memchr::memchr(b'"', body.as_bytes()).is_some() {
        1
    } else {
        n_chunks
    };
    let bounds = chunk_bounds(body.as_bytes(), n_chunks);
    let texts: Vec<&str> = bounds
        .windows(2)
        .map(|w| body.get(w[0]..w[1]).unwrap_or(""))
        .collect();

    // Pass 1: tokenize chunks to spans.
    let results: Vec<Result<ChunkSpans>> = texts
        .par_iter()
        .map(|t| tokenize_chunk(t, n_fields, &field_to_out, n_out))
        .collect();
    let mut chunks = Vec::with_capacity(results.len());
    let mut bases = Vec::with_capacity(results.len());
    let mut rows_before = 0usize;
    for r in results {
        match r {
            Ok(c) => {
                bases.push(rows_before);
                rows_before += c.rows;
                chunks.push(c);
            }
            Err(e) => return Err(shift(e, rows_before + 1)),
        }
    }

    // Pass 2: per-(chunk, column) inference over the inference window, merged per column.
    let limit = opts.infer_rows.unwrap_or(usize::MAX);
    let tasks: Vec<(usize, usize)> = (0..n_out)
        .flat_map(|c| (0..chunks.len()).map(move |k| (c, k)))
        .collect();
    let flags: Vec<Flags> = tasks
        .par_iter()
        .map(|&(c, k)| {
            let spans = &chunks[k].cols[c];
            let take = limit.saturating_sub(bases[k]).min(spans.len());
            infer_chunk(texts[k], &spans[..take])
        })
        .collect();
    let types: Vec<DataType> = flags
        .chunks(chunks.len())
        .map(|fs| fs.iter().fold(Flags::START, |a, b| a.merge(*b)).data_type())
        .collect();

    // Pass 3: parse each (chunk, column) straight into a typed builder.
    let built: Vec<Result<Array>> = tasks
        .par_iter()
        .map(|&(c, k)| build_chunk(texts[k], &chunks[k].cols[c], types[c], &header[outputs[c]]))
        .collect();
    let mut per_col: Vec<Vec<Array>> = (0..n_out).map(|_| Vec::new()).collect();
    let mut first_err: Option<CsvError> = None;
    for (&(c, k), r) in tasks.iter().zip(built) {
        match r {
            Ok(a) => per_col[c].push(a),
            Err(e) => {
                let e = shift(e, bases[k] + 1);
                if first_err
                    .as_ref()
                    .is_none_or(|f| record_of(&e) < record_of(f))
                {
                    first_err = Some(e);
                }
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }

    // Concatenate chunk arrays per column.
    let cols = per_col
        .into_par_iter()
        .zip(types.par_iter())
        .enumerate()
        .map(|(c, (parts, dt))| Ok(Series::new(header[outputs[c]].clone(), concat(parts, *dt)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(DataFrame::new(cols)?)
}

fn write_field(out: &mut String, s: &str, force_quote: bool) {
    if force_quote || s.contains([',', '"', '\n', '\r']) {
        out.push('"');
        out.push_str(&s.replace('"', "\"\""));
        out.push('"');
    } else {
        out.push_str(s);
    }
}

/// Serialise a [`DataFrame`] to CSV text (header + `\n`-terminated rows).
pub fn write_csv_string(df: &DataFrame) -> String {
    let mut out = String::new();
    for (i, c) in df.columns().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_field(&mut out, c.name(), c.name().is_empty());
    }
    out.push('\n');
    for r in 0..df.height() {
        for (i, c) in df.columns().iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            match c.get(r) {
                Scalar::Null(_) => {}
                Scalar::Utf8(s) => write_field(&mut out, &s, s.is_empty()),
                Scalar::Float64(v) => {
                    let _ = write!(out, "{v:?}");
                }
                other => {
                    let _ = write!(out, "{other}");
                }
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_and_nulls() {
        let csv = "a,b,c,d\n1,1.5,true,x\n,2,false,\n3,,,\"\"\n";
        let df = read_csv_bytes(csv.as_bytes()).unwrap();
        let t: Vec<_> = df.columns().iter().map(|c| c.data_type()).collect();
        assert_eq!(
            t,
            vec![
                DataType::Int64,
                DataType::Float64,
                DataType::Boolean,
                DataType::Utf8
            ]
        );
        assert_eq!(
            df.column("a").unwrap().get(1),
            Scalar::Null(DataType::Int64)
        );
        assert_eq!(df.column("d").unwrap().get(1), Scalar::Null(DataType::Utf8));
        assert_eq!(df.column("d").unwrap().get(2), Scalar::Utf8(String::new()));
    }

    #[test]
    fn quotes_and_crlf() {
        let csv = "name,v\r\n\"a,\"\"b\"\"\nc\",1\r\nplain,2";
        let df = read_csv_bytes(csv.as_bytes()).unwrap();
        assert_eq!(
            df.column("name").unwrap().get(0),
            Scalar::Utf8("a,\"b\"\nc".into())
        );
        assert_eq!(df.height(), 2);
        assert!(matches!(
            read_csv_bytes(b"a\n\"x"),
            Err(CsvError::UnterminatedQuote(_))
        ));
        assert!(matches!(
            read_csv_bytes(b"a,b\n1"),
            Err(CsvError::FieldCount { .. })
        ));
        assert!(matches!(read_csv_bytes(b""), Err(CsvError::Empty)));
        assert!(matches!(
            read_csv_bytes(b"a\n\"x\"y"),
            Err(CsvError::BadQuote(_))
        ));
    }
}
