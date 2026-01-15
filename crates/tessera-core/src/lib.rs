//! # tessera-core
//!
//! [`Series`] (a named [`Array`]) and [`DataFrame`] (equal-length series).
//! Row filtering runs over columns in parallel with rayon.
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod groupby;
pub(crate) mod keys;
pub mod ops;

use std::fmt;

use rayon::prelude::*;
use tessera_arrow::{Array, ArrowError, BooleanArray, BooleanBuilder, MutableBitmap, Utf8Builder};
use tessera_compute::filter::{filter_f64, filter_primitive};
use tessera_compute::take::{take_boolean, take_primitive, take_utf8};
use tessera_compute::ComputeError;
use thiserror::Error;

pub use groupby::AggKind;
pub use ops::{merged_key_type, JoinType};
pub use tessera_arrow::{DataType, Scalar};

/// Errors from DataFrame operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CoreError {
    /// Columns of different lengths.
    #[error("column {0:?} has length {1}, expected {2}")]
    LengthMismatch(String, usize, usize),
    /// Unknown column name.
    #[error("column not found: {0:?}")]
    ColumnNotFound(String),
    /// Two columns share a name.
    #[error("duplicate column name: {0:?}")]
    Duplicate(String),
    /// Kernel failure.
    #[error(transparent)]
    Compute(#[from] ComputeError),
    /// Array construction failure.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Row indices are `u32`, so frames and gathers are limited to `u32::MAX` rows.
    #[error("row index {0} does not fit in u32")]
    TooManyRows(usize),
    /// Invalid arguments (empty key list, mismatched flag count, ...).
    #[error("invalid argument: {0}")]
    Invalid(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, CoreError>;

/// A named column.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    name: String,
    array: Array,
}

impl Series {
    /// Create a series.
    pub fn new(name: impl Into<String>, array: Array) -> Self {
        Series {
            name: name.into(),
            array,
        }
    }
    /// Name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Underlying array.
    pub fn array(&self) -> &Array {
        &self.array
    }
    /// Length.
    pub fn len(&self) -> usize {
        self.array.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }
    /// Data type.
    pub fn data_type(&self) -> DataType {
        self.array.data_type()
    }
    /// Value at row `i`.
    pub fn get(&self, i: usize) -> Scalar {
        self.array.scalar(i)
    }
    /// Keep rows where `mask` is true (null = drop).
    pub fn filter(&self, mask: &BooleanArray) -> Result<Series> {
        let array = match &self.array {
            Array::Int32(a) => Array::Int32(filter_primitive(a, mask)?),
            Array::Int64(a) => Array::Int64(filter_primitive(a, mask)?),
            Array::Float64(a) => Array::Float64(filter_f64(a, mask)?),
            Array::Boolean(a) => {
                check_len(&self.name, mask.len(), a.len())?;
                let mut b = BooleanBuilder::new();
                for i in selected(mask) {
                    b.push(a.get(i));
                }
                Array::Boolean(b.finish())
            }
            Array::Utf8(a) => {
                check_len(&self.name, mask.len(), a.len())?;
                let mut b = Utf8Builder::new();
                for i in selected(mask) {
                    b.push(a.get(i));
                }
                Array::Utf8(b.finish()?)
            }
        };
        Ok(Series::new(self.name.clone(), array))
    }
    /// Zero-copy slice.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Series> {
        Ok(Series::new(
            self.name.clone(),
            self.array.slice(offset, len)?,
        ))
    }
    /// Gather rows by index (an out-of-range index is an error).
    pub fn take(&self, idx: &[usize]) -> Result<Series> {
        let idx = idx
            .iter()
            .map(|&i| {
                u32::try_from(i)
                    .map(Some)
                    .map_err(|_| CoreError::TooManyRows(i))
            })
            .collect::<Result<Vec<_>>>()?;
        self.take_opt(&idx)
    }
    /// Typed gather with optional indices (`None` yields a null), using the
    /// `tessera_compute::take` kernels.
    pub fn take_opt(&self, idx: &[Option<u32>]) -> Result<Series> {
        let array = match &self.array {
            Array::Int32(a) => Array::Int32(take_primitive(a, idx)?),
            Array::Int64(a) => Array::Int64(take_primitive(a, idx)?),
            Array::Float64(a) => Array::Float64(take_primitive(a, idx)?),
            Array::Boolean(a) => Array::Boolean(take_boolean(a, idx)?),
            Array::Utf8(a) => Array::Utf8(take_utf8(a, idx)?),
        };
        Ok(Series::new(self.name.clone(), array))
    }
}

fn check_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(CoreError::LengthMismatch(name.to_owned(), got, want));
    }
    Ok(())
}

fn selected(mask: &BooleanArray) -> impl Iterator<Item = usize> + '_ {
    (0..mask.len()).filter(move |&i| mask.get(i) == Some(true))
}

/// A table of equal-length named columns.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DataFrame {
    columns: Vec<Series>,
    height: usize,
}

impl DataFrame {
    /// Create, validating equal lengths and unique names.
    pub fn new(columns: Vec<Series>) -> Result<Self> {
        let height = columns.first().map_or(0, Series::len);
        for (i, c) in columns.iter().enumerate() {
            check_len(c.name(), c.len(), height)?;
            if columns[..i].iter().any(|o| o.name() == c.name()) {
                return Err(CoreError::Duplicate(c.name().to_owned()));
            }
        }
        Ok(DataFrame { columns, height })
    }
    /// (rows, columns).
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.columns.len())
    }
    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }
    /// All columns.
    pub fn columns(&self) -> &[Series] {
        &self.columns
    }
    /// Column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(Series::name).collect()
    }
    /// Column by name.
    pub fn column(&self, name: &str) -> Result<&Series> {
        self.columns
            .iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| CoreError::ColumnNotFound(name.to_owned()))
    }
    /// Add a column, or replace the existing column of the same name in place.
    /// The series must have the frame's height (any length is accepted when
    /// the frame has no columns yet).
    pub fn with_column(&self, series: Series) -> Result<DataFrame> {
        let mut cols = self.columns.clone();
        match cols.iter().position(|c| c.name() == series.name()) {
            Some(i) if cols.len() == 1 => cols[i] = series,
            Some(i) => {
                check_len(series.name(), series.len(), self.height)?;
                cols[i] = series;
            }
            None if cols.is_empty() => cols.push(series),
            None => {
                check_len(series.name(), series.len(), self.height)?;
                cols.push(series);
            }
        }
        DataFrame::new(cols)
    }
    /// Project columns by name (in the given order).
    pub fn select(&self, names: &[&str]) -> Result<DataFrame> {
        let cols = names
            .iter()
            .map(|n| self.column(n).cloned())
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(cols)
    }
    /// Keep rows where `mask` is true; columns are filtered in parallel.
    pub fn filter(&self, mask: &BooleanArray) -> Result<DataFrame> {
        check_len("<mask>", mask.len(), self.height)?;
        let cols = self
            .columns
            .par_iter()
            .map(|c| c.filter(mask))
            .collect::<Result<Vec<_>>>()?;
        let height = mask.validity().map_or_else(
            || mask.values().count_set(),
            |v| mask.values().and(v).map_or(0, |b| b.count_set()),
        );
        Ok(DataFrame {
            columns: cols,
            height,
        })
    }
    /// First `n` rows (zero-copy).
    pub fn head(&self, n: usize) -> Result<DataFrame> {
        let n = n.min(self.height);
        let cols = self
            .columns
            .iter()
            .map(|c| c.slice(0, n))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFrame {
            columns: cols,
            height: n,
        })
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MAX_ROWS: usize = 10;
        let shown = self.height.min(MAX_ROWS);
        let mut cells: Vec<Vec<String>> = self
            .columns
            .iter()
            .map(|c| {
                let mut v = vec![c.name().to_owned(), c.data_type().to_string()];
                v.extend((0..shown).map(|i| c.get(i).to_string()));
                v
            })
            .collect();
        if shown < self.height {
            for c in &mut cells {
                c.push("…".into());
            }
        }
        let widths: Vec<usize> = cells
            .iter()
            .map(|c| c.iter().map(|s| s.chars().count()).max().unwrap_or(0))
            .collect();
        let sep: String = widths
            .iter()
            .map(|w| format!("+{}", "-".repeat(w + 2)))
            .collect::<String>()
            + "+";
        writeln!(f, "shape: ({}, {})", self.height, self.columns.len())?;
        writeln!(f, "{sep}")?;
        let rows = cells.first().map_or(0, Vec::len);
        for r in 0..rows {
            for (c, w) in cells.iter().zip(&widths) {
                write!(f, "| {:<w$} ", c[r], w = w)?;
            }
            writeln!(f, "|")?;
            if r == 1 {
                writeln!(f, "{sep}")?;
            }
        }
        write!(f, "{sep}")
    }
}

/// Build a validity-carrying boolean mask from bools (helper for callers).
pub fn mask_from_bools(bits: impl IntoIterator<Item = bool>) -> BooleanArray {
    let mut m = MutableBitmap::new();
    for b in bits {
        m.push(b);
    }
    BooleanArray::try_new(m.into_bitmap(), None).unwrap_or_else(|_| BooleanArray::from_options([]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_arrow::PrimitiveArray;
    use tessera_arrow::Utf8Array;

    fn df() -> DataFrame {
        DataFrame::new(vec![
            Series::new(
                "a",
                Array::Int64(PrimitiveArray::from_options([
                    Some(1),
                    None,
                    Some(3),
                    Some(4),
                ])),
            ),
            Series::new(
                "b",
                Array::Utf8(
                    Utf8Array::from_options([Some("x"), Some("y"), None, Some("z")]).unwrap(),
                ),
            ),
            Series::new(
                "c",
                Array::Float64(PrimitiveArray::from_options([
                    Some(0.5),
                    Some(1.5),
                    Some(2.5),
                    None,
                ])),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn construct_select_shape() {
        let d = df();
        assert_eq!(d.shape(), (4, 3));
        assert_eq!(
            d.select(&["c", "a"]).unwrap().column_names(),
            vec!["c", "a"]
        );
        assert!(matches!(
            d.select(&["zz"]),
            Err(CoreError::ColumnNotFound(_))
        ));
        let bad = DataFrame::new(vec![
            Series::new("a", Array::Int64(PrimitiveArray::from_options([Some(1)]))),
            Series::new(
                "b",
                Array::Int64(PrimitiveArray::from_options([Some(1), Some(2)])),
            ),
        ]);
        assert!(matches!(bad, Err(CoreError::LengthMismatch(..))));
        let dup = DataFrame::new(vec![d.columns()[0].clone(), d.columns()[0].clone()]);
        assert!(matches!(dup, Err(CoreError::Duplicate(_))));
    }

    #[test]
    fn filter_and_head() {
        let d = df();
        let m = BooleanArray::from_options([Some(true), Some(true), None, Some(false)]);
        let f = d.filter(&m).unwrap();
        assert_eq!(f.shape(), (2, 3));
        assert_eq!(f.column("a").unwrap().get(1), Scalar::Null(DataType::Int64));
        assert_eq!(f.column("b").unwrap().get(1), Scalar::Utf8("y".into()));
        assert_eq!(d.head(2).unwrap().height(), 2);
        assert_eq!(d.head(100).unwrap().height(), 4);
        assert!(d.filter(&mask_from_bools([true])).is_err());
    }

    #[test]
    fn display() {
        let s = df().to_string();
        assert!(s.starts_with("shape: (4, 3)"));
        assert!(s.contains("| a    | b    |"), "{s}");
        assert!(s.contains("null"));
    }
}
