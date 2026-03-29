//! Element-wise arithmetic. Integer ops wrap; validity is the AND of inputs.
//! Integer division by zero yields null (floats follow IEEE-754).
use tessera_arrow::{Bitmap, MutableBitmap, NativeType, PrimitiveArray};

use crate::error::{ComputeError, Result};

/// Apply a binary op over two equal-length arrays (auto-vectorized loop).
pub fn binary<T: NativeType>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    op: impl Fn(T, T) -> T,
) -> Result<PrimitiveArray<T>> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch(a.len(), b.len()));
    }
    let values = a
        .values()
        .iter()
        .zip(b.values().iter())
        .map(|(x, y)| op(*x, *y))
        .collect();
    let validity = match (a.validity(), b.validity()) {
        (Some(p), Some(q)) => Some(p.and(q)?),
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (None, None) => None,
    };
    Ok(PrimitiveArray::try_new(values, validity)?)
}

/// `a + b` for i64 (wrapping).
pub fn add_i64(a: &PrimitiveArray<i64>, b: &PrimitiveArray<i64>) -> Result<PrimitiveArray<i64>> {
    binary(a, b, i64::wrapping_add)
}
/// `a * b` for i64 (wrapping).
pub fn mul_i64(a: &PrimitiveArray<i64>, b: &PrimitiveArray<i64>) -> Result<PrimitiveArray<i64>> {
    binary(a, b, i64::wrapping_mul)
}
/// `a + b` for f64.
pub fn add_f64(a: &PrimitiveArray<f64>, b: &PrimitiveArray<f64>) -> Result<PrimitiveArray<f64>> {
    binary(a, b, |x, y| x + y)
}

/// `a - b` for i32 (wrapping).
pub fn sub_i32(a: &PrimitiveArray<i32>, b: &PrimitiveArray<i32>) -> Result<PrimitiveArray<i32>> {
    binary(a, b, i32::wrapping_sub)
}
/// `a - b` for i64 (wrapping).
pub fn sub_i64(a: &PrimitiveArray<i64>, b: &PrimitiveArray<i64>) -> Result<PrimitiveArray<i64>> {
    binary(a, b, i64::wrapping_sub)
}
/// `a - b` for f64.
pub fn sub_f64(a: &PrimitiveArray<f64>, b: &PrimitiveArray<f64>) -> Result<PrimitiveArray<f64>> {
    binary(a, b, |x, y| x - y)
}
/// `a / b` for f64 (IEEE-754: division by zero gives inf/NaN, not null).
pub fn div_f64(a: &PrimitiveArray<f64>, b: &PrimitiveArray<f64>) -> Result<PrimitiveArray<f64>> {
    binary(a, b, |x, y| x / y)
}

/// Integer division: rows where the divisor is zero become null. Wrapping
/// for `MIN / -1`.
fn div_checked<T: NativeType>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    zero: T,
    op: impl Fn(T, T) -> T,
) -> Result<PrimitiveArray<T>> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch(a.len(), b.len()));
    }
    let (x, y) = (a.values(), b.values());
    let mut nonzero = MutableBitmap::with_capacity(x.len());
    let mut any_zero = false;
    let values = x
        .iter()
        .zip(y.iter())
        .map(|(p, q)| {
            let ok = *q != zero;
            nonzero.push(ok);
            any_zero |= !ok;
            if ok {
                op(*p, *q)
            } else {
                zero
            }
        })
        .collect();
    let base = crate::util::and_validity(a.validity(), b.validity())?;
    let validity: Option<Bitmap> = match (base, any_zero) {
        (Some(v), true) => Some(v.and(&nonzero.into_bitmap())?),
        (Some(v), false) => Some(v),
        (None, true) => Some(nonzero.into_bitmap()),
        (None, false) => None,
    };
    Ok(PrimitiveArray::try_new(values, validity)?)
}

/// `a / b` for i32; null where `b == 0` (wrapping `MIN / -1`).
pub fn div_i32(a: &PrimitiveArray<i32>, b: &PrimitiveArray<i32>) -> Result<PrimitiveArray<i32>> {
    div_checked(a, b, 0, i32::wrapping_div)
}
/// `a / b` for i64; null where `b == 0` (wrapping `MIN / -1`).
pub fn div_i64(a: &PrimitiveArray<i64>, b: &PrimitiveArray<i64>) -> Result<PrimitiveArray<i64>> {
    div_checked(a, b, 0, i64::wrapping_div)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_propagates_nulls_and_wraps() {
        let a = PrimitiveArray::from_options([Some(i64::MAX), None, Some(2)]);
        let b = PrimitiveArray::from_options([Some(1), Some(1), None]);
        let r = add_i64(&a, &b).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(i64::MIN), None, None]
        );
        assert!(add_i64(&a, &PrimitiveArray::from_options([Some(1)])).is_err());
    }

    #[test]
    fn sub_wraps_and_propagates_nulls() {
        let a = PrimitiveArray::from_options([Some(i64::MIN), None, Some(5)]);
        let b = PrimitiveArray::from_options([Some(1), Some(1), Some(7)]);
        let r = sub_i64(&a, &b).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(i64::MAX), None, Some(-2)]
        );
        let x = PrimitiveArray::from_options([Some(1.5), Some(2.0)]);
        assert_eq!(
            sub_f64(&x, &x).unwrap().iter().collect::<Vec<_>>(),
            vec![Some(0.0), Some(0.0)]
        );
        let i = PrimitiveArray::from_options([Some(i32::MIN), Some(3)]);
        let j = PrimitiveArray::from_options([Some(1), Some(1)]);
        assert_eq!(
            sub_i32(&i, &j).unwrap().iter().collect::<Vec<_>>(),
            vec![Some(i32::MAX), Some(2)]
        );
    }

    #[test]
    fn int_div_by_zero_is_null_float_is_ieee() {
        let a = PrimitiveArray::from_options([Some(7i64), Some(7), None, Some(i64::MIN), Some(9)]);
        let b = PrimitiveArray::from_options([Some(2i64), Some(0), Some(1), Some(-1), None]);
        let r = div_i64(&a, &b).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(3), None, None, Some(i64::MIN), None]
        );
        // no zero, no nulls: validity stays absent
        let c = PrimitiveArray::from_options([Some(8i32), Some(9)]);
        let d = PrimitiveArray::from_options([Some(2i32), Some(3)]);
        let r = div_i32(&c, &d).unwrap();
        assert!(r.validity().is_none());
        assert_eq!(r.values().as_slice(), &[4, 3]);
        let f = PrimitiveArray::from_options([Some(1.0), Some(0.0)]);
        let z = PrimitiveArray::from_options([Some(0.0), Some(0.0)]);
        let r = div_f64(&f, &z).unwrap();
        assert_eq!(r.get(0), Some(f64::INFINITY));
        assert!(r.get(1).is_some_and(f64::is_nan));
        assert!(div_i64(&a, &PrimitiveArray::from_options([Some(1)])).is_err());
    }

    #[test]
    fn sliced_division() {
        let a = PrimitiveArray::from_options((0..10i64).map(|i| (i != 3).then_some(i * 10)));
        let b = PrimitiveArray::from_options((0..10i64).map(|i| Some(i % 3)));
        let r = div_i64(&a.slice(2, 6).unwrap(), &b.slice(2, 6).unwrap()).unwrap();
        // rows 2..8: 20/2, None, 40/1, 50/2, 60/0, 70/1
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(10), None, Some(40), Some(25), None, Some(70)]
        );
    }
}
