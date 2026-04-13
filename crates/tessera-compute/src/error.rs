//! Kernel errors.
use thiserror::Error;

/// Errors raised by compute kernels.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// Input arrays have different lengths.
    #[error("length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),
    /// Input has more rows than a `u32` index can address.
    #[error("{0} rows exceed the u32 index range")]
    TooLong(usize),
    /// Underlying array construction error.
    #[error(transparent)]
    Arrow(#[from] tessera_arrow::ArrowError),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, ComputeError>;
