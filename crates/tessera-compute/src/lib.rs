//! # tessera-compute
//!
//! Vectorized kernels over [`tessera_arrow`] arrays.
//!
//! Each hot kernel exists in up to three flavours so they can be compared:
//! * `*_scalar` — the obvious one-element-at-a-time loop (reference);
//! * `*_chunked` — fixed-width chunked loops with independent accumulators
//!   that LLVM auto-vectorizes (null handling via 64-bit bitmap words and
//!   branch-free selects);
//! * `*_avx2` — explicit `std::arch` AVX2 intrinsics, selected at runtime
//!   with `is_x86_feature_detected!`; non-x86 or non-AVX2 CPUs fall back to
//!   the chunked path.
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

pub mod aggregate;
pub mod arithmetic;
pub mod boolean;
pub mod cast;
pub mod comparison;
pub mod error;
pub mod filter;
pub mod hash;
pub mod sort;
pub mod strings;
pub mod take;

mod util;

pub use error::{ComputeError, Result};

/// Whether the AVX2 code paths are usable on this CPU.
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
