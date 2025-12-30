//! # tessera-arrow
//!
//! A from-scratch implementation of the [Apache Arrow columnar format]
//! physical memory layout for a small set of types.
//!
//! ## Which parts of the Arrow spec are followed
//!
//! | Spec item | Status |
//! |---|---|
//! | Buffers allocated on 64-byte aligned addresses | yes for every non-empty allocation ([`buffer::MutableBuffer`] uses `Layout` align 64) |
//! | Buffer capacity padded to a multiple of 64 bytes | yes (allocation size is rounded up) |
//! | Validity bitmap, LSB bit numbering, `1` = valid | yes ([`bitmap::Bitmap`]) |
//! | Validity bitmap may be omitted when there are no nulls | yes (`Option<Bitmap>`) |
//! | Fixed-size primitive layout (`Int32`, `Int64`, `Float64`) | yes, little-endian native |
//! | Boolean layout: values are a bitmap | yes ([`array::BooleanArray`]) |
//! | Variable-size binary `Utf8`: `i32` offsets buffer of length `n+1`, monotonically non-decreasing, plus data buffer | yes ([`array::Utf8Array`]) |
//! | Logical array `offset` for zero-copy slicing | yes — buffers and bitmaps carry an element/bit offset |
//! | Value slots behind a null are "undefined" | builders write zeros, readers never rely on them |
//! | `LargeUtf8`, nested, dictionary, temporal, decimal, IPC, C Data Interface | **not implemented** |
//!
//! [Apache Arrow columnar format]: https://arrow.apache.org/docs/format/Columnar.html
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod array;
pub mod bitmap;
pub mod buffer;
pub mod builder;
pub mod datatype;
pub mod error;

pub use array::{Array, BooleanArray, PrimitiveArray, Utf8Array};
pub use bitmap::{Bitmap, MutableBitmap};
pub use buffer::{Buffer, MutableBuffer, NativeType, ALIGNMENT};
pub use builder::{BooleanBuilder, PrimitiveBuilder, Utf8Builder};
pub use datatype::{DataType, Scalar};
pub use error::{ArrowError, Result};
