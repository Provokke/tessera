//! Logical data types and scalar values.
use std::fmt;

/// The data types supported by tessera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Bit-packed booleans.
    Boolean,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// IEEE-754 double.
    Float64,
    /// UTF-8 strings with `i32` offsets.
    Utf8,
}

impl DataType {
    /// Whether the type is numeric (integer or float).
    pub fn is_numeric(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64 | DataType::Float64)
    }
    /// Whether the type is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Boolean => "bool",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float64 => "f64",
            DataType::Utf8 => "str",
        };
        f.write_str(s)
    }
}

/// A single, possibly-null value.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// A typed null.
    Null(DataType),
    /// Boolean.
    Boolean(bool),
    /// Int32.
    Int32(i32),
    /// Int64.
    Int64(i64),
    /// Float64.
    Float64(f64),
    /// String.
    Utf8(String),
}

impl Scalar {
    /// The data type of the scalar.
    pub fn data_type(&self) -> DataType {
        match self {
            Scalar::Null(dt) => *dt,
            Scalar::Boolean(_) => DataType::Boolean,
            Scalar::Int32(_) => DataType::Int32,
            Scalar::Int64(_) => DataType::Int64,
            Scalar::Float64(_) => DataType::Float64,
            Scalar::Utf8(_) => DataType::Utf8,
        }
    }
    /// Whether the scalar is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Scalar::Null(_))
    }
    /// Numeric value widened to f64, if numeric and non-null.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Scalar::Int32(v) => Some(*v as f64),
            Scalar::Int64(v) => Some(*v as f64),
            Scalar::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Null(_) => f.write_str("null"),
            Scalar::Boolean(v) => write!(f, "{v}"),
            Scalar::Int32(v) => write!(f, "{v}"),
            Scalar::Int64(v) => write!(f, "{v}"),
            Scalar::Float64(v) => write!(f, "{v:?}"),
            Scalar::Utf8(v) => write!(f, "{v:?}"),
        }
    }
}
