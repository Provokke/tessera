//! Comparison kernels producing [`BooleanArray`]s, packed 64 lanes at a time.
use tessera_arrow::{Bitmap, BooleanArray, NativeType, PrimitiveArray, Utf8Array};

use crate::error::{ComputeError, Result};

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
}

#[inline]
fn apply<T: PartialOrd>(op: CmpOp, a: T, b: T) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::NotEq => a != b,
        CmpOp::Lt => a < b,
        CmpOp::LtEq => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::GtEq => a >= b,
    }
}

fn pack(len: usize, f: impl Fn(usize) -> bool) -> Bitmap {
    let mut words = Vec::with_capacity(len.div_ceil(64));
    for base in (0..len).step_by(64) {
        let n = (len - base).min(64);
        let mut w = 0u64;
        for j in 0..n {
            w |= (f(base + j) as u64) << j;
        }
        words.push(w);
    }
    Bitmap::from_words(&words, len)
}

/// Compare every element against a scalar; validity is carried over.
pub fn cmp_scalar<T: NativeType + PartialOrd>(
    a: &PrimitiveArray<T>,
    op: CmpOp,
    rhs: T,
) -> BooleanArray {
    let v = a.values();
    let bits = pack(v.len(), |i| apply(op, v[i], rhs));
    BooleanArray::try_new(bits, a.validity().cloned())
        .unwrap_or_else(|_| BooleanArray::from_options([]))
}

/// Element-wise comparison of two arrays; result is null where either is null.
pub fn cmp<T: NativeType + PartialOrd>(
    a: &PrimitiveArray<T>,
    op: CmpOp,
    b: &PrimitiveArray<T>,
) -> Result<BooleanArray> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch(a.len(), b.len()));
    }
    let (x, y) = (a.values(), b.values());
    let bits = pack(x.len(), |i| apply(op, x[i], y[i]));
    let validity = match (a.validity(), b.validity()) {
        (Some(p), Some(q)) => Some(p.and(q)?),
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (None, None) => None,
    };
    Ok(BooleanArray::try_new(bits, validity)?)
}

/// Compare every string against a scalar (byte-wise lexicographic order);
/// validity is carried over.
pub fn cmp_utf8_scalar(a: &Utf8Array, op: CmpOp, rhs: &str) -> Result<BooleanArray> {
    let bits = pack(a.len(), |i| apply(op, a.value(i), rhs));
    Ok(BooleanArray::try_new(bits, a.validity().cloned())?)
}

/// Element-wise string comparison; result is null where either side is null.
pub fn cmp_utf8(a: &Utf8Array, op: CmpOp, b: &Utf8Array) -> Result<BooleanArray> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch(a.len(), b.len()));
    }
    let bits = pack(a.len(), |i| apply(op, a.value(i), b.value(i)));
    let validity = crate::util::and_validity(a.validity(), b.validity())?;
    Ok(BooleanArray::try_new(bits, validity)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_array_cmp_with_nulls() {
        let a = PrimitiveArray::from_options([Some(1i64), None, Some(3)]);
        let r = cmp_scalar(&a, CmpOp::Gt, 1);
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(false), None, Some(true)]
        );
        let b = PrimitiveArray::from_options([Some(1i64), Some(2), None]);
        let r = cmp(&a, CmpOp::Eq, &b).unwrap();
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![Some(true), None, None]);
    }

    #[test]
    fn utf8_comparisons_with_nulls_and_slices() {
        let a =
            Utf8Array::from_options([Some("b"), None, Some("a"), Some("abc"), Some("")]).unwrap();
        let r = cmp_utf8_scalar(&a, CmpOp::Gt, "ab").unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(false), Some(true), Some(false)]
        );
        let b =
            Utf8Array::from_options([Some("b"), Some("x"), None, Some("abd"), Some("")]).unwrap();
        let r = cmp_utf8(&a, CmpOp::Eq, &b).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(true), None, None, Some(false), Some(true)]
        );
        let r = cmp_utf8(&a.slice(2, 3).unwrap(), CmpOp::Lt, &b.slice(2, 3).unwrap()).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![None, Some(true), Some(false)]
        );
        let r = cmp_utf8_scalar(&a.slice(3, 2).unwrap(), CmpOp::NotEq, "").unwrap();
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![Some(true), Some(false)]);
        assert!(cmp_utf8(&a, CmpOp::Eq, &b.slice(1, 2).unwrap()).is_err());
    }
}
