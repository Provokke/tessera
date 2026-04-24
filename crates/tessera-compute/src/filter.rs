//! Filter kernels: keep rows where the mask is `true` (null mask = drop).
use tessera_arrow::{
    Bitmap, BooleanArray, MutableBitmap, MutableBuffer, NativeType, PrimitiveArray, Utf8Array,
    Utf8Builder,
};

use crate::error::{ComputeError, Result};

/// Effective selection bitmap: `values AND validity`.
fn selection(mask: &BooleanArray) -> Result<Bitmap> {
    Ok(match mask.validity() {
        Some(v) => mask.values().and(v)?,
        None => mask.values().clone(),
    })
}

/// Scalar reference: per-element branch.
pub fn filter_slice_scalar<T: NativeType>(v: &[T], sel: &Bitmap) -> Vec<T> {
    v.iter()
        .enumerate()
        .filter(|(i, _)| sel.get(*i))
        .map(|(_, x)| *x)
        .collect()
}

/// Word-at-a-time filter: full words are bulk-copied, empty words skipped,
/// mixed words walked with `trailing_zeros`.
pub fn filter_slice_chunked<T: NativeType>(v: &[T], sel: &Bitmap) -> MutableBuffer<T> {
    let mut out = MutableBuffer::with_capacity(sel.count_set());
    for (ci, chunk) in v.chunks(64).enumerate() {
        let mut w = sel.word(ci * 64);
        if w == u64::MAX {
            out.extend_from_slice(chunk);
            continue;
        }
        while w != 0 {
            let j = w.trailing_zeros() as usize;
            out.push(chunk[j]);
            w &= w - 1;
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// For each 4-bit lane mask, the 32-bit lane permutation that packs the
    /// selected 64-bit lanes to the front.
    const fn build_lut() -> [[i32; 8]; 16] {
        let mut lut = [[0i32; 8]; 16];
        let mut m = 0;
        while m < 16 {
            let mut k = 0;
            let mut lane: i32 = 0;
            while lane < 4 {
                if m & (1usize << lane) != 0 {
                    lut[m][2 * k] = 2 * lane;
                    lut[m][2 * k + 1] = 2 * lane + 1;
                    k += 1;
                }
                lane += 1;
            }
            m += 1;
        }
        lut
    }
    static LUT: [[i32; 8]; 16] = build_lut();

    /// Compress-store of 64-bit lanes. `out` must have room for
    /// `selected + 4` elements; returns number written.
    ///
    /// # Safety
    /// CPU must support AVX2; `out.len() >= popcount(sel) + 4`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_u64(v: &[u64], words: &[u64], out: &mut [u64]) -> usize {
        let mut n = 0usize;
        for (ci, chunk) in v.chunks(64).enumerate() {
            let w = words[ci];
            let quads = chunk.chunks_exact(4);
            let rem_start = chunk.len() - quads.remainder().len();
            for (qi, q) in quads.enumerate() {
                let m = ((w >> (qi * 4)) & 0xF) as usize;
                if m == 0 {
                    continue;
                }
                // SAFETY: `q` has 4 u64s; LUT row has 8 i32s; `out` has at
                // least 4 free slots past `n` because n <= popcount so far.
                unsafe {
                    let x = _mm256_loadu_si256(q.as_ptr().cast());
                    let p = _mm256_loadu_si256(LUT[m].as_ptr().cast());
                    let packed = _mm256_permutevar8x32_epi32(x, p);
                    _mm256_storeu_si256(out.as_mut_ptr().add(n).cast(), packed);
                }
                n += m.count_ones() as usize;
            }
            for (j, x) in chunk.iter().enumerate().skip(rem_start) {
                if (w >> j) & 1 == 1 {
                    out[n] = *x;
                    n += 1;
                }
            }
        }
        n
    }
}

/// AVX2 compress-store filter for 8-byte values (`u64` bit patterns).
/// Falls back to the chunked kernel without AVX2.
pub fn filter_u64_avx2(v: &[u64], sel: &Bitmap) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    if crate::avx2_available() {
        let words: Vec<u64> = sel.words().collect();
        let mut out = vec![0u64; sel.count_set() + 4];
        // SAFETY: AVX2 verified; `out` has popcount + 4 slots; `words`
        // covers every 64-element chunk of `v` (sel.len() == v.len()).
        let n = unsafe { avx2::filter_u64(v, &words, &mut out) };
        out.truncate(n);
        return out;
    }
    filter_slice_chunked(v, sel).as_slice().to_vec()
}

fn filter_validity(validity: Option<&Bitmap>, sel: &Bitmap) -> Option<Bitmap> {
    let v = validity?;
    let mut out = MutableBitmap::with_capacity(sel.count_set());
    for i in 0..sel.len() {
        if sel.get(i) {
            out.push(v.get(i));
        }
    }
    Some(out.into_bitmap())
}

/// Filter a primitive array by a boolean mask.
pub fn filter_primitive<T: NativeType>(
    a: &PrimitiveArray<T>,
    mask: &BooleanArray,
) -> Result<PrimitiveArray<T>> {
    if a.len() != mask.len() {
        return Err(ComputeError::LengthMismatch(a.len(), mask.len()));
    }
    let sel = selection(mask)?;
    let values = filter_slice_chunked(a.values(), &sel).into_buffer();
    Ok(PrimitiveArray::try_new(
        values,
        filter_validity(a.validity(), &sel),
    )?)
}

/// Filter an f64 array using the AVX2 compress path when available.
pub fn filter_f64(a: &PrimitiveArray<f64>, mask: &BooleanArray) -> Result<PrimitiveArray<f64>> {
    if a.len() != mask.len() {
        return Err(ComputeError::LengthMismatch(a.len(), mask.len()));
    }
    let sel = selection(mask)?;
    let bits: Vec<u64> = a.values().iter().map(|x| x.to_bits()).collect();
    let values = filter_u64_avx2(&bits, &sel)
        .into_iter()
        .map(f64::from_bits)
        .collect();
    Ok(PrimitiveArray::try_new(
        values,
        filter_validity(a.validity(), &sel),
    )?)
}

/// Filter a boolean array by a boolean mask.
pub fn filter_boolean(a: &BooleanArray, mask: &BooleanArray) -> Result<BooleanArray> {
    if a.len() != mask.len() {
        return Err(ComputeError::LengthMismatch(a.len(), mask.len()));
    }
    let sel = selection(mask)?;
    let mut values = MutableBitmap::with_capacity(sel.count_set());
    for i in 0..sel.len() {
        if sel.get(i) {
            values.push(a.values().get(i));
        }
    }
    Ok(BooleanArray::try_new(
        values.into_bitmap(),
        filter_validity(a.validity(), &sel),
    )?)
}

/// Filter a string array by a boolean mask.
pub fn filter_utf8(a: &Utf8Array, mask: &BooleanArray) -> Result<Utf8Array> {
    if a.len() != mask.len() {
        return Err(ComputeError::LengthMismatch(a.len(), mask.len()));
    }
    let sel = selection(mask)?;
    let mut b = Utf8Builder::new();
    for i in 0..sel.len() {
        if sel.get(i) {
            b.push(a.get(i));
        }
    }
    Ok(b.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_with_null_mask_and_values() {
        let a = PrimitiveArray::from_options([Some(1i64), None, Some(3), Some(4), Some(5)]);
        let m = BooleanArray::from_options([Some(true), Some(true), None, Some(false), Some(true)]);
        let f = filter_primitive(&a, &m).unwrap();
        assert_eq!(f.iter().collect::<Vec<_>>(), vec![Some(1), None, Some(5)]);
        assert!(filter_primitive(&a, &BooleanArray::from_options([Some(true)])).is_err());
    }

    #[test]
    fn avx2_matches_scalar() {
        let v: Vec<u64> = (0..1000).collect();
        let sel = Bitmap::from_bools((0..1000).map(|i| (i * 13) % 7 < 3));
        assert_eq!(filter_u64_avx2(&v, &sel), filter_slice_scalar(&v, &sel));
        let f = PrimitiveArray::from_options((0..77).map(|i| Some(i as f64)));
        let m = BooleanArray::from_options((0..77).map(|i| Some(i % 2 == 0)));
        assert_eq!(filter_f64(&f, &m).unwrap().len(), 39);
    }

    #[test]
    fn filter_utf8_and_boolean_sliced() {
        let u =
            Utf8Array::from_options([Some("a"), None, Some("bb"), Some("ccc"), None, Some("d")])
                .unwrap();
        let m = BooleanArray::from_options([
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(true),
        ]);
        let r = filter_utf8(&u, &m).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some("a"), None, None, Some("d")]
        );
        let (su, sm) = (u.slice(2, 4).unwrap(), m.slice(2, 4).unwrap());
        let r = filter_utf8(&su, &sm).unwrap();
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![None, Some("d")]);
        let b = BooleanArray::from_options([
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ]);
        let r = filter_boolean(&b.slice(2, 4).unwrap(), &sm).unwrap();
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![None, Some(false)]);
        assert!(filter_boolean(&b, &sm).is_err());
        assert!(filter_utf8(&u, &sm).is_err());
    }
}
