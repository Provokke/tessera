//! Shared plumbing for hash group-by and hash join: typed key-column
//! equality and parallel per-row hashing.
use rayon::prelude::*;
use tessera_arrow::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use tessera_compute::hash::hash_columns;

use crate::{CoreError, Result};

/// A borrowed key column with typed, allocation-free row equality.
pub(crate) enum KeyCol<'a> {
    I32(&'a PrimitiveArray<i32>),
    I64(&'a PrimitiveArray<i64>),
    F64(&'a PrimitiveArray<f64>),
    Bool(&'a BooleanArray),
    Str(&'a Utf8Array),
}

/// Canonical float key: `-0.0 == 0.0`; other values compare by bit pattern
/// (so NaNs with equal bits are equal, matching the reference group-by).
fn canon(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

impl<'a> KeyCol<'a> {
    pub(crate) fn new(a: &'a Array) -> KeyCol<'a> {
        match a {
            Array::Int32(x) => KeyCol::I32(x),
            Array::Int64(x) => KeyCol::I64(x),
            Array::Float64(x) => KeyCol::F64(x),
            Array::Boolean(x) => KeyCol::Bool(x),
            Array::Utf8(x) => KeyCol::Str(x),
        }
    }

    pub(crate) fn is_null(&self, i: usize) -> bool {
        match self {
            KeyCol::I32(a) => !a.is_valid(i),
            KeyCol::I64(a) => !a.is_valid(i),
            KeyCol::F64(a) => !a.is_valid(i),
            KeyCol::Bool(a) => a.get(i).is_none(),
            KeyCol::Str(a) => a.get(i).is_none(),
        }
    }

    /// `Some(value)` for the integer kinds, `None` for the others.
    fn as_int(&self, i: usize) -> Option<Option<i64>> {
        match self {
            KeyCol::I32(a) => Some(a.get(i).map(i64::from)),
            KeyCol::I64(a) => Some(a.get(i)),
            _ => None,
        }
    }

    /// Row `i` of `self` equals row `j` of `o`. Two nulls are equal (callers
    /// that need SQL semantics skip null rows first). Integer widths compare
    /// by value; other kind mismatches are never equal.
    pub(crate) fn eq(&self, i: usize, o: &KeyCol<'_>, j: usize) -> bool {
        match (self, o) {
            (KeyCol::I64(a), KeyCol::I64(b)) => a.get(i) == b.get(j),
            (KeyCol::F64(a), KeyCol::F64(b)) => a.get(i).map(canon) == b.get(j).map(canon),
            (KeyCol::Bool(a), KeyCol::Bool(b)) => a.get(i) == b.get(j),
            (KeyCol::Str(a), KeyCol::Str(b)) => a.get(i) == b.get(j),
            _ => match (self.as_int(i), o.as_int(j)) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            },
        }
    }
}

/// Row `a` equals row `b` on every key column of the same frame.
pub(crate) fn rows_eq(keys: &[KeyCol<'_>], a: usize, b: usize) -> bool {
    keys.iter().all(|k| k.eq(a, k, b))
}

/// Check the row count fits the `u32` row ids used by the hash operators.
pub(crate) fn check_rows(n: usize) -> Result<()> {
    u32::try_from(n)
        .map(|_| ())
        .map_err(|_| CoreError::TooManyRows(n))
}

const HASH_CHUNK: usize = 1 << 17;

/// Per-row combined hashes of `cols` (all of length `n`), computed over row
/// chunks in parallel. With no columns every row hashes to 0.
pub(crate) fn hash_rows(cols: &[&Array], n: usize) -> Result<Vec<u64>> {
    if cols.is_empty() {
        return Ok(vec![0; n]);
    }
    if n <= HASH_CHUNK {
        return Ok(hash_columns(cols)?);
    }
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(HASH_CHUNK)
        .map(|s| (s, HASH_CHUNK.min(n - s)))
        .collect();
    let parts = ranges
        .par_iter()
        .map(|&(s, l)| -> Result<Vec<u64>> {
            let sl = cols
                .iter()
                .map(|c| c.slice(s, l))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let refs: Vec<&Array> = sl.iter().collect();
            Ok(hash_columns(&refs)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend_from_slice(&p);
    }
    Ok(out)
}
