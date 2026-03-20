//! Sum / min / max reductions (scalar, auto-vectorized, AVX2).
//!
//! Semantics: nulls are skipped; `sum` of an all-null or empty array is
//! `None`; float `min`/`max` additionally ignore NaN (like pandas `skipna`)
//! and return `None` when no non-NaN valid value exists.
use tessera_arrow::{Bitmap, PrimitiveArray};

const LANES: usize = 8;

/// Reference scalar sum of a slice.
pub fn sum_f64_scalar(v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for x in v {
        acc += *x;
    }
    acc
}

/// Auto-vectorizable sum: 8 independent accumulators over fixed chunks.
pub fn sum_f64_chunked(v: &[f64]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let chunks = v.chunks_exact(LANES);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..LANES {
            acc[i] += c[i];
        }
    }
    acc.iter().sum::<f64>() + rem.iter().sum::<f64>()
}

/// Auto-vectorizable i64 sum (wrapping).
pub fn sum_i64_chunked(v: &[i64]) -> i64 {
    let mut acc = [0i64; LANES];
    let chunks = v.chunks_exact(LANES);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..LANES {
            acc[i] = acc[i].wrapping_add(c[i]);
        }
    }
    acc.iter().chain(rem).fold(0i64, |a, b| a.wrapping_add(*b))
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// # Safety
    /// Caller must ensure the CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_f64(v: &[f64]) -> f64 {
        let chunks = v.chunks_exact(8);
        let rem = chunks.remainder();
        let mut a = _mm256_setzero_pd();
        let mut b = _mm256_setzero_pd();
        for c in chunks {
            // SAFETY: `c` has exactly 8 f64s; loadu has no alignment requirement.
            unsafe {
                a = _mm256_add_pd(a, _mm256_loadu_pd(c.as_ptr()));
                b = _mm256_add_pd(b, _mm256_loadu_pd(c.as_ptr().add(4)));
            }
        }
        let mut out = [0.0f64; 4];
        // SAFETY: `out` has room for 4 f64s.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), _mm256_add_pd(a, b)) };
        out.iter().sum::<f64>() + rem.iter().sum::<f64>()
    }

    /// # Safety
    /// Caller must ensure the CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_i64(v: &[i64]) -> i64 {
        let chunks = v.chunks_exact(4);
        let rem = chunks.remainder();
        let mut a = _mm256_setzero_si256();
        for c in chunks {
            // SAFETY: `c` has exactly 4 i64s; loadu is unaligned-safe.
            unsafe { a = _mm256_add_epi64(a, _mm256_loadu_si256(c.as_ptr().cast())) };
        }
        let mut out = [0i64; 4];
        // SAFETY: `out` has room for 4 i64s.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), a) };
        out.iter().chain(rem).fold(0i64, |x, y| x.wrapping_add(*y))
    }

    /// NaN-ignoring min (`MIN = true`) or max of a slice; returns ±inf sentinel.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn minmax_f64<const MIN: bool>(v: &[f64]) -> f64 {
        let init = if MIN {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let mut acc = _mm256_set1_pd(init);
        let chunks = v.chunks_exact(4);
        let rem = chunks.remainder();
        for c in chunks {
            // SAFETY: 4 f64s available; unaligned load.
            let x = unsafe { _mm256_loadu_pd(c.as_ptr()) };
            // Ordered comparisons are false for NaN, so NaN is never selected.
            let take = if MIN {
                _mm256_cmp_pd::<_CMP_LT_OQ>(x, acc)
            } else {
                _mm256_cmp_pd::<_CMP_GT_OQ>(x, acc)
            };
            acc = _mm256_blendv_pd(acc, x, take);
        }
        let mut out = [0.0f64; 4];
        // SAFETY: room for 4 f64s.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), acc) };
        out.iter()
            .chain(rem)
            .fold(init, |a, &x| super::pick_f64::<MIN>(a, x))
    }

    /// Min/max of i64 using compare + blend (AVX2 lacks `min_epi64`).
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn minmax_i64<const MIN: bool>(v: &[i64]) -> i64 {
        let init = if MIN { i64::MAX } else { i64::MIN };
        let mut acc = _mm256_set1_epi64x(init);
        let chunks = v.chunks_exact(4);
        let rem = chunks.remainder();
        for c in chunks {
            // SAFETY: 4 i64s available; unaligned load.
            let x = unsafe { _mm256_loadu_si256(c.as_ptr().cast()) };
            let take = if MIN {
                _mm256_cmpgt_epi64(acc, x)
            } else {
                _mm256_cmpgt_epi64(x, acc)
            };
            acc = _mm256_blendv_epi8(acc, x, take);
        }
        let mut out = [0i64; 4];
        // SAFETY: room for 4 i64s.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), acc) };
        out.iter()
            .chain(rem)
            .fold(init, |a, &x| if MIN { a.min(x) } else { a.max(x) })
    }
}

#[inline]
fn pick_f64<const MIN: bool>(acc: f64, x: f64) -> f64 {
    let take = if MIN { x < acc } else { x > acc };
    if take {
        x
    } else {
        acc
    }
}

/// Explicit AVX2 sum; falls back to the chunked kernel without AVX2.
pub fn sum_f64_avx2(v: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::avx2_available() {
        // SAFETY: AVX2 support was verified at runtime.
        return unsafe { avx2::sum_f64(v) };
    }
    sum_f64_chunked(v)
}

/// Explicit AVX2 i64 sum with fallback.
pub fn sum_i64_avx2(v: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if crate::avx2_available() {
        // SAFETY: AVX2 support was verified at runtime.
        return unsafe { avx2::sum_i64(v) };
    }
    sum_i64_chunked(v)
}

fn minmax_f64_slice<const MIN: bool>(v: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::avx2_available() {
        // SAFETY: AVX2 support was verified at runtime.
        return unsafe { avx2::minmax_f64::<MIN>(v) };
    }
    let init = if MIN {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    v.iter().fold(init, |a, &x| pick_f64::<MIN>(a, x))
}

fn minmax_i64_slice<const MIN: bool>(v: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if crate::avx2_available() {
        // SAFETY: AVX2 support was verified at runtime.
        return unsafe { avx2::minmax_i64::<MIN>(v) };
    }
    let init = if MIN { i64::MAX } else { i64::MIN };
    v.iter()
        .fold(init, |a, &x| if MIN { a.min(x) } else { a.max(x) })
}

/// Masked sum over values whose validity bit is set, 64 at a time with a
/// branch-free select so the inner loop vectorizes.
fn masked_sum_f64(v: &[f64], validity: &Bitmap) -> f64 {
    let mut total = 0.0;
    for (ci, chunk) in v.chunks(64).enumerate() {
        let w = validity.word(ci * 64);
        let mut acc = [0.0f64; LANES];
        for (j, x) in chunk.iter().enumerate() {
            let keep = (w >> j) & 1 == 1;
            acc[j % LANES] += if keep { *x } else { 0.0 };
        }
        total += acc.iter().sum::<f64>();
    }
    total
}

/// Sum of a float array (nulls skipped). `None` if no valid values.
pub fn sum_f64(a: &PrimitiveArray<f64>) -> Option<f64> {
    if a.len() == a.null_count() {
        return None;
    }
    Some(match a.validity() {
        Some(v) if v.unset_bits() > 0 => masked_sum_f64(a.values(), v),
        _ => sum_f64_avx2(a.values()),
    })
}

/// Sum of an i64 array (wrapping, nulls skipped). `None` if no valid values.
pub fn sum_i64(a: &PrimitiveArray<i64>) -> Option<i64> {
    if a.len() == a.null_count() {
        return None;
    }
    Some(match a.validity() {
        Some(v) if v.unset_bits() > 0 => {
            let mut total = 0i64;
            for (ci, chunk) in a.values().chunks(64).enumerate() {
                let w = validity_word(v, ci);
                for (j, x) in chunk.iter().enumerate() {
                    // Mask is all-ones or zero: branch-free and vectorizable.
                    let m = 0i64.wrapping_sub(((w >> j) & 1) as i64);
                    total = total.wrapping_add(x & m);
                }
            }
            total
        }
        _ => sum_i64_avx2(a.values()),
    })
}

#[inline]
fn validity_word(v: &Bitmap, chunk: usize) -> u64 {
    v.word(chunk * 64)
}

fn minmax_f64<const MIN: bool>(a: &PrimitiveArray<f64>) -> Option<f64> {
    let sentinel = if MIN {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let r = match a.validity() {
        Some(v) if v.unset_bits() > 0 => a.iter().flatten().fold(sentinel, pick_f64::<MIN>),
        _ => minmax_f64_slice::<MIN>(a.values()),
    };
    if r == sentinel && !a.iter().flatten().any(|x| x == sentinel) {
        None
    } else {
        Some(r)
    }
}

fn minmax_i64<const MIN: bool>(a: &PrimitiveArray<i64>) -> Option<i64> {
    if a.len() == a.null_count() {
        return None;
    }
    match a.validity() {
        Some(v) if v.unset_bits() > 0 => {
            let it = a.iter().flatten();
            if MIN {
                it.min()
            } else {
                it.max()
            }
        }
        _ => Some(minmax_i64_slice::<MIN>(a.values())),
    }
}

/// Minimum of a float array, ignoring nulls and NaN.
pub fn min_f64(a: &PrimitiveArray<f64>) -> Option<f64> {
    minmax_f64::<true>(a)
}
/// Maximum of a float array, ignoring nulls and NaN.
pub fn max_f64(a: &PrimitiveArray<f64>) -> Option<f64> {
    minmax_f64::<false>(a)
}
/// Minimum of an i64 array, ignoring nulls.
pub fn min_i64(a: &PrimitiveArray<i64>) -> Option<i64> {
    minmax_i64::<true>(a)
}
/// Maximum of an i64 array, ignoring nulls.
pub fn max_i64(a: &PrimitiveArray<i64>) -> Option<i64> {
    minmax_i64::<false>(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_with_nulls() {
        let a = PrimitiveArray::from_options([Some(1.5), None, Some(2.5), None]);
        assert_eq!(sum_f64(&a), Some(4.0));
        let b = PrimitiveArray::<f64>::from_options([None, None]);
        assert_eq!(sum_f64(&b), None);
        let e = PrimitiveArray::<i64>::from_options([]);
        assert_eq!(sum_i64(&e), None);
        let c = PrimitiveArray::from_options((0..200i64).map(|i| (i % 7 != 0).then_some(i)));
        assert_eq!(sum_i64(&c), Some((0..200).filter(|i| i % 7 != 0).sum()));
    }

    #[test]
    fn minmax_nan_and_nulls() {
        let a = PrimitiveArray::from_options([
            Some(f64::NAN),
            Some(3.0),
            None,
            Some(-1.0),
            Some(f64::NAN),
        ]);
        assert_eq!(min_f64(&a), Some(-1.0));
        assert_eq!(max_f64(&a), Some(3.0));
        let n = PrimitiveArray::from_options([Some(f64::NAN), None]);
        assert_eq!(min_f64(&n), None);
        let inf = PrimitiveArray::from_options([Some(f64::INFINITY)]);
        assert_eq!(min_f64(&inf), Some(f64::INFINITY));
        let i = PrimitiveArray::from_options([Some(5i64), None, Some(i64::MIN), Some(9)]);
        assert_eq!(min_i64(&i), Some(i64::MIN));
        assert_eq!(max_i64(&i), Some(9));
        assert_eq!(max_i64(&PrimitiveArray::from_options([None::<i64>])), None);
    }

    #[test]
    fn sliced_inputs() {
        let a = PrimitiveArray::from_options((0..300i64).map(|i| (i % 5 != 0).then_some(i)));
        let s = a.slice(3, 250).unwrap();
        let expect: i64 = (3..253).filter(|i| i % 5 != 0).sum();
        assert_eq!(sum_i64(&s), Some(expect));
        assert_eq!(min_i64(&s), Some(3));
        assert_eq!(max_i64(&s), Some(252));
    }
}
