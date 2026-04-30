//! Stable argsort kernels returning `u32` row indices (apply with
//! [`crate::take`]). Nulls always sort last, also when `descending`.
//!
//! Integer and float keys are mapped to order-preserving unsigned keys and
//! sorted with an LSD radix sort (8 bits per pass, constant-digit passes are
//! skipped). Descending order inverts the key, so ties keep input order in
//! both directions. Floats use the IEEE-754 total order: `-NaN < -inf < ... <
//! -0.0 < +0.0 < ... < +inf < +NaN`. Strings use a stable comparison sort over
//! their UTF-8 bytes.
use tessera_arrow::{PrimitiveArray, Utf8Array};

use crate::error::{ComputeError, Result};

fn check_len(len: usize) -> Result<()> {
    if u32::try_from(len).is_err() {
        return Err(ComputeError::TooLong(len));
    }
    Ok(())
}

/// Stable LSD radix sort of `(key, index)` pairs on the low `bits` bits.
fn radix_sort(mut items: Vec<(u64, u32)>, bits: u32) -> Vec<(u64, u32)> {
    let mut tmp = vec![(0u64, 0u32); items.len()];
    for pass in 0..bits / 8 {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        for (k, _) in &items {
            counts[((k >> shift) & 0xFF) as usize] += 1;
        }
        // Every key has the same digit: this pass cannot reorder anything.
        if counts.contains(&items.len()) {
            continue;
        }
        let mut pos = [0usize; 256];
        let mut total = 0;
        for (p, c) in pos.iter_mut().zip(counts) {
            *p = total;
            total += c;
        }
        for it in &items {
            let d = ((it.0 >> shift) & 0xFF) as usize;
            tmp[pos[d]] = *it;
            pos[d] += 1;
        }
        std::mem::swap(&mut items, &mut tmp);
    }
    items
}

/// Split rows into (valid, null) and radix-sort the valid ones.
fn argsort_keys(
    len: usize,
    is_valid: impl Fn(usize) -> bool,
    key: impl Fn(usize) -> u64,
    bits: u32,
    descending: bool,
) -> Result<Vec<u32>> {
    check_len(len)?;
    let mut items = Vec::with_capacity(len);
    let mut nulls = Vec::new();
    for i in 0..len {
        if is_valid(i) {
            let k = key(i);
            items.push((if descending { !k } else { k }, i as u32));
        } else {
            nulls.push(i as u32);
        }
    }
    let mut out: Vec<u32> = radix_sort(items, bits)
        .into_iter()
        .map(|(_, i)| i)
        .collect();
    out.extend(nulls);
    Ok(out)
}

/// Stable argsort of an `i32` array.
pub fn argsort_i32(a: &PrimitiveArray<i32>, descending: bool) -> Result<Vec<u32>> {
    let v = a.values();
    argsort_keys(
        a.len(),
        |i| a.is_valid(i),
        |i| u64::from((v[i] as u32) ^ 0x8000_0000),
        32,
        descending,
    )
}

/// Stable argsort of an `i64` array.
pub fn argsort_i64(a: &PrimitiveArray<i64>, descending: bool) -> Result<Vec<u32>> {
    let v = a.values();
    argsort_keys(
        a.len(),
        |i| a.is_valid(i),
        |i| (v[i] as u64) ^ (1 << 63),
        64,
        descending,
    )
}

/// Map an `f64` to a `u64` whose unsigned order is the IEEE-754 total order.
fn f64_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

/// Stable argsort of an `f64` array (IEEE-754 total order, see module docs).
pub fn argsort_f64(a: &PrimitiveArray<f64>, descending: bool) -> Result<Vec<u32>> {
    let v = a.values();
    argsort_keys(
        a.len(),
        |i| a.is_valid(i),
        |i| f64_key(v[i]),
        64,
        descending,
    )
}

/// Stable argsort of a string array (byte-wise lexicographic order).
pub fn argsort_utf8(a: &Utf8Array, descending: bool) -> Result<Vec<u32>> {
    check_len(a.len())?;
    let mut idx = Vec::with_capacity(a.len());
    let mut nulls = Vec::new();
    for i in 0..a.len() {
        if a.get(i).is_some() {
            idx.push(i as u32);
        } else {
            nulls.push(i as u32);
        }
    }
    if descending {
        idx.sort_by(|x, y| a.value(*y as usize).cmp(a.value(*x as usize)));
    } else {
        idx.sort_by(|x, y| a.value(*x as usize).cmp(a.value(*y as usize)));
    }
    idx.extend(nulls);
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_stable_nulls_last_descending() {
        let a = PrimitiveArray::from_options([
            Some(3i64),
            None,
            Some(-1),
            Some(3),
            Some(i64::MIN),
            None,
            Some(i64::MAX),
        ]);
        assert_eq!(argsort_i64(&a, false).unwrap(), vec![4, 2, 0, 3, 6, 1, 5]);
        // ties (0 and 3) keep input order when descending as well
        assert_eq!(argsort_i64(&a, true).unwrap(), vec![6, 0, 3, 2, 4, 1, 5]);
    }

    #[test]
    fn i32_and_sliced() {
        let a = PrimitiveArray::from_options((0..10i32).map(|i| (i % 4 != 0).then_some(5 - i)));
        let s = a.slice(2, 7).unwrap();
        let got = argsort_i32(&s, false).unwrap();
        let mut want: Vec<u32> = (0..7).filter(|i| s.get(*i as usize).is_some()).collect();
        want.sort_by_key(|i| s.get(*i as usize));
        want.extend((0..7).filter(|i| s.get(*i as usize).is_none()));
        assert_eq!(got, want);
        assert!(argsort_i32(&PrimitiveArray::<i32>::from_options([]), true)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn f64_total_order() {
        let a = PrimitiveArray::from_options([
            Some(1.5),
            Some(f64::NAN),
            Some(-0.0),
            Some(0.0),
            Some(f64::NEG_INFINITY),
            None,
            Some(f64::INFINITY),
            Some(-2.0),
        ]);
        assert_eq!(
            argsort_f64(&a, false).unwrap(),
            vec![4, 7, 2, 3, 0, 6, 1, 5]
        );
        assert_eq!(argsort_f64(&a, true).unwrap(), vec![1, 6, 0, 3, 2, 7, 4, 5]);
    }

    #[test]
    fn utf8_sort() {
        let a =
            Utf8Array::from_options([Some("b"), None, Some("a"), Some("b"), Some(""), Some("ab")])
                .unwrap();
        assert_eq!(argsort_utf8(&a, false).unwrap(), vec![4, 2, 5, 0, 3, 1]);
        assert_eq!(argsort_utf8(&a, true).unwrap(), vec![0, 3, 5, 2, 4, 1]);
        let s = a.slice(1, 4).unwrap();
        assert_eq!(argsort_utf8(&s, false).unwrap(), vec![3, 1, 2, 0]);
    }
}
