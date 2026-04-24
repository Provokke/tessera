//! Per-row 64-bit hashing and cross-column hash combination.
//!
//! Hashes are deterministic across runs (no random seed). Null rows hash to
//! the constant [`NULL_HASH`] regardless of what the slot physically holds.
//! `Int32` rows hash like the equal `Int64` value, so mixed-width integer
//! keys agree. Floats are canonicalised so `-0.0 == 0.0` and every NaN hash
//! alike.
use tessera_arrow::{Array, BooleanArray, NativeType, PrimitiveArray, Utf8Array};

use crate::error::{ComputeError, Result};

/// Hash of every null row.
pub const NULL_HASH: u64 = 0x9E37_79B9_7F4A_7C15;

const MUL: u64 = 0xBF58_476D_1CE4_E5B9;

/// splitmix64-style finaliser.
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(MUL);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn hash_u64(x: u64) -> u64 {
    mix(x.wrapping_add(0x2545_F491_4F6C_DD1D))
}

#[inline]
fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = 0x243F_6A88_85A3_08D3u64 ^ (b.len() as u64);
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        let mut w = [0u8; 8];
        w.copy_from_slice(c);
        h = mix(h ^ u64::from_le_bytes(w));
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let mut w = [0u8; 8];
        w[..rest.len()].copy_from_slice(rest);
        h = mix(h ^ u64::from_le_bytes(w));
    }
    h
}

fn hash_prim<T: NativeType>(a: &PrimitiveArray<T>, f: impl Fn(T) -> u64) -> Vec<u64> {
    let v = a.values();
    match a.validity() {
        None => v.iter().map(|x| f(*x)).collect(),
        Some(valid) => v
            .iter()
            .enumerate()
            .map(|(i, x)| if valid.get(i) { f(*x) } else { NULL_HASH })
            .collect(),
    }
}

/// Hash an `i32` array (same hashes as the equal `i64` values).
pub fn hash_i32(a: &PrimitiveArray<i32>) -> Vec<u64> {
    hash_prim(a, |x| hash_u64(i64::from(x) as u64))
}

/// Hash an `i64` array.
pub fn hash_i64(a: &PrimitiveArray<i64>) -> Vec<u64> {
    hash_prim(a, |x| hash_u64(x as u64))
}

/// Hash an `f64` array (`-0.0` and `0.0` agree; all NaNs agree).
pub fn hash_f64(a: &PrimitiveArray<f64>) -> Vec<u64> {
    hash_prim(a, |x| {
        let bits = if x.is_nan() {
            f64::NAN.to_bits()
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        };
        hash_u64(bits ^ 0xF10A7)
    })
}

/// Hash a boolean array.
pub fn hash_boolean(a: &BooleanArray) -> Vec<u64> {
    (0..a.len())
        .map(|i| match a.get(i) {
            Some(b) => hash_u64(u64::from(b) ^ 0xB001),
            None => NULL_HASH,
        })
        .collect()
}

/// Hash a string array.
pub fn hash_utf8(a: &Utf8Array) -> Vec<u64> {
    (0..a.len())
        .map(|i| match a.get(i) {
            Some(s) => hash_bytes(s.as_bytes()),
            None => NULL_HASH,
        })
        .collect()
}

/// Hash any array.
pub fn hash_array(a: &Array) -> Vec<u64> {
    match a {
        Array::Boolean(x) => hash_boolean(x),
        Array::Int32(x) => hash_i32(x),
        Array::Int64(x) => hash_i64(x),
        Array::Float64(x) => hash_f64(x),
        Array::Utf8(x) => hash_utf8(x),
    }
}

/// Fold `other` into `acc` element-wise. Order-sensitive: combining column
/// A then B differs from B then A.
pub fn combine_hashes(acc: &mut [u64], other: &[u64]) -> Result<()> {
    if acc.len() != other.len() {
        return Err(ComputeError::LengthMismatch(acc.len(), other.len()));
    }
    for (a, o) in acc.iter_mut().zip(other) {
        *a = mix(a.rotate_left(23) ^ o).wrapping_add(MUL);
    }
    Ok(())
}

/// Row hashes over several equal-length columns. With no columns the result
/// is empty.
pub fn hash_columns(cols: &[&Array]) -> Result<Vec<u64>> {
    let mut it = cols.iter();
    let Some(first) = it.next() else {
        return Ok(Vec::new());
    };
    let mut acc = hash_array(first);
    for c in it {
        combine_hashes(&mut acc, &hash_array(c))?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nulls_are_constant_and_equal_values_agree() {
        let a = PrimitiveArray::from_options([Some(5i64), None, Some(5), None, Some(6)]);
        let h = hash_i64(&a);
        assert_eq!(h[0], h[2]);
        assert_ne!(h[0], h[4]);
        assert_eq!(h[1], NULL_HASH);
        assert_eq!(h[3], NULL_HASH);
        let b = PrimitiveArray::from_options([Some(5i32), None, Some(6)]);
        let hb = hash_i32(&b);
        assert_eq!((hb[0], hb[1], hb[2]), (h[0], NULL_HASH, h[4]));
    }

    #[test]
    fn sliced_equals_unsliced() {
        let u = Utf8Array::from_options([
            Some("alpha"),
            None,
            Some("a considerably longer string"),
            Some(""),
            Some("alpha"),
        ])
        .unwrap();
        let full = hash_utf8(&u);
        assert_eq!(hash_utf8(&u.slice(1, 3).unwrap()), full[1..4]);
        assert_eq!(full[0], full[4]);
        assert_eq!(full[1], NULL_HASH);
        let b = BooleanArray::from_options([Some(true), None, Some(false), Some(true)]);
        let hb = hash_boolean(&b);
        assert_eq!(hash_boolean(&b.slice(2, 2).unwrap()), hb[2..]);
        assert_eq!(hb[0], hb[3]);
        assert_ne!(hb[0], hb[2]);
    }

    #[test]
    fn float_canonicalisation() {
        let a =
            PrimitiveArray::from_options([Some(0.0), Some(-0.0), Some(f64::NAN), Some(-f64::NAN)]);
        let h = hash_f64(&a);
        assert_eq!(h[0], h[1]);
        assert_eq!(h[2], h[3]);
        assert_ne!(h[0], h[2]);
    }

    #[test]
    fn combine_is_order_sensitive_and_checks_length() {
        let x = Array::Int64(PrimitiveArray::from_options([Some(1), Some(2), None]));
        let y = Array::Utf8(Utf8Array::from_options([Some("a"), None, Some("c")]).unwrap());
        let xy = hash_columns(&[&x, &y]).unwrap();
        let yx = hash_columns(&[&y, &x]).unwrap();
        assert_ne!(xy, yx);
        assert_eq!(xy.len(), 3);
        assert!(hash_columns(&[]).unwrap().is_empty());
        let short = Array::Int64(PrimitiveArray::from_options([Some(1)]));
        assert!(hash_columns(&[&x, &short]).is_err());
    }
}
