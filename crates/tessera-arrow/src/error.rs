//! Error type for array construction and validation.
use thiserror::Error;

/// Errors raised when constructing or validating arrays.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ArrowError {
    /// A buffer/bitmap length does not match the array length.
    #[error("length mismatch: {0}")]
    LengthMismatch(String),
    /// Offsets are not monotonically non-decreasing or exceed the data buffer.
    #[error("invalid offsets: {0}")]
    InvalidOffsets(String),
    /// String data is not valid UTF-8.
    #[error("invalid utf-8 in string data")]
    InvalidUtf8,
    /// Data exceeds what an `i32` offsets buffer can address.
    #[error("utf8 data exceeds i32::MAX bytes")]
    OffsetOverflow,
    /// Out-of-bounds slice or index.
    #[error("out of bounds: {0}")]
    OutOfBounds(String),
    /// The requested allocation could not be satisfied.
    #[error("allocation of {0} bytes failed")]
    Alloc(usize),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, ArrowError>;
