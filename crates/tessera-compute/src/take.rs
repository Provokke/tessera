//! Gather kernels: `out[i] = values[indices[i]]`, with `None` indices
//! producing nulls. Out-of-bounds indices are an error.
use tessera_arrow::{
    ArrowError, BooleanArray, MutableBitmap, MutableBuffer, NativeType, PrimitiveArray, Utf8Array,
    Utf8Builder,
};

use crate::error::Result;

fn check_bounds(indices: &[Option<u32>], len: usize) -> Result<()> {
    match indices.iter().flatten().find(|i| **i as usize >= len) {
        Some(i) => Err(ArrowError::OutOfBounds(format!("take index {i} of {len}")).into()),
        None => Ok(()),
    }
}

/// Take from a primitive array.
pub fn take_primitive<T: NativeType>(
    a: &PrimitiveArray<T>,
    indices: &[Option<u32>],
) -> Result<PrimitiveArray<T>> {
    check_bounds(indices, a.len())?;
    let src = a.values();
    let need_validity = a.validity().is_some() || indices.iter().any(Option::is_none);
    let mut values = MutableBuffer::with_capacity(indices.len());
    let mut validity = MutableBitmap::with_capacity(if need_validity { indices.len() } else { 0 });
    for idx in indices {
        match idx {
            Some(k) => {
                let k = *k as usize;
                values.push(src[k]);
                if need_validity {
                    validity.push(a.is_valid(k));
                }
            }
            None => {
                values.push(T::default());
                validity.push(false);
            }
        }
    }
    Ok(PrimitiveArray::try_new(
        values.into_buffer(),
        need_validity.then(|| validity.into_bitmap()),
    )?)
}

/// Take from a boolean array.
pub fn take_boolean(a: &BooleanArray, indices: &[Option<u32>]) -> Result<BooleanArray> {
    check_bounds(indices, a.len())?;
    let need_validity = a.validity().is_some() || indices.iter().any(Option::is_none);
    let mut values = MutableBitmap::with_capacity(indices.len());
    let mut validity = MutableBitmap::with_capacity(if need_validity { indices.len() } else { 0 });
    for idx in indices {
        match idx {
            Some(k) => {
                let k = *k as usize;
                values.push(a.values().get(k));
                if need_validity {
                    validity.push(a.get(k).is_some());
                }
            }
            None => {
                values.push(false);
                validity.push(false);
            }
        }
    }
    Ok(BooleanArray::try_new(
        values.into_bitmap(),
        need_validity.then(|| validity.into_bitmap()),
    )?)
}

/// Take from a string array.
pub fn take_utf8(a: &Utf8Array, indices: &[Option<u32>]) -> Result<Utf8Array> {
    check_bounds(indices, a.len())?;
    let mut b = Utf8Builder::new();
    for idx in indices {
        b.push(idx.and_then(|k| a.get(k as usize)));
    }
    Ok(b.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_with_null_indices_and_values() {
        let a = PrimitiveArray::from_options([Some(10i64), None, Some(30), Some(40)]);
        let r = take_primitive(&a, &[Some(3), None, Some(1), Some(0)]).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(40), None, None, Some(10)]
        );
        // No nulls anywhere: no validity bitmap allocated.
        let b = PrimitiveArray::from_options([Some(1i32), Some(2)]);
        let r = take_primitive(&b, &[Some(1), Some(1), Some(0)]).unwrap();
        assert!(r.validity().is_none());
        assert_eq!(r.values().as_slice(), &[2, 2, 1]);
        assert!(take_primitive(&b, &[Some(2)]).is_err());
        assert!(take_primitive(&b, &[]).unwrap().is_empty());
    }

    #[test]
    fn sliced_inputs() {
        let a = PrimitiveArray::from_options((0..20i64).map(|i| (i % 4 != 0).then_some(i)));
        let s = a.slice(5, 10).unwrap();
        let r = take_primitive(&s, &[Some(0), Some(3), Some(9), None]).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(5), None, Some(14), None]
        );
        let u =
            Utf8Array::from_options([Some("a"), None, Some("ccc"), Some("dd"), Some("e")]).unwrap();
        let s = u.slice(1, 4).unwrap();
        let r = take_utf8(&s, &[Some(3), Some(0), Some(1), None]).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some("e"), None, Some("ccc"), None]
        );
        assert!(take_utf8(&s, &[Some(4)]).is_err());
    }

    #[test]
    fn boolean_take() {
        let a = BooleanArray::from_options([Some(true), None, Some(false), Some(true)]);
        let s = a.slice(1, 3).unwrap();
        let r = take_boolean(&s, &[Some(2), Some(0), None, Some(1)]).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(true), None, None, Some(false)]
        );
        assert!(take_boolean(&s, &[Some(3)]).is_err());
    }
}
