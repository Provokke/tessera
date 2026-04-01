//! Boolean kernels. Nulls propagate (no Kleene logic): the result is null
//! wherever any input is null.
use tessera_arrow::BooleanArray;

use crate::error::{ComputeError, Result};
use crate::util::and_validity;

fn check(a: &BooleanArray, b: &BooleanArray) -> Result<()> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch(a.len(), b.len()));
    }
    Ok(())
}

/// Element-wise `a AND b`; null where either side is null.
pub fn and(a: &BooleanArray, b: &BooleanArray) -> Result<BooleanArray> {
    check(a, b)?;
    let values = a.values().and(b.values())?;
    Ok(BooleanArray::try_new(
        values,
        and_validity(a.validity(), b.validity())?,
    )?)
}

/// Element-wise `a OR b`; null where either side is null.
pub fn or(a: &BooleanArray, b: &BooleanArray) -> Result<BooleanArray> {
    check(a, b)?;
    let values = a.values().or(b.values())?;
    Ok(BooleanArray::try_new(
        values,
        and_validity(a.validity(), b.validity())?,
    )?)
}

/// Element-wise `NOT a`; validity is carried over.
pub fn not(a: &BooleanArray) -> Result<BooleanArray> {
    Ok(BooleanArray::try_new(
        a.values().not(),
        a.validity().cloned(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(a: &BooleanArray) -> Vec<Option<bool>> {
        a.iter().collect()
    }

    #[test]
    fn and_or_not_propagate_nulls() {
        let a = BooleanArray::from_options([Some(true), Some(false), None, Some(true)]);
        let b = BooleanArray::from_options([Some(true), None, Some(false), Some(false)]);
        assert_eq!(
            v(&and(&a, &b).unwrap()),
            vec![Some(true), None, None, Some(false)]
        );
        assert_eq!(
            v(&or(&a, &b).unwrap()),
            vec![Some(true), None, None, Some(true)]
        );
        assert_eq!(
            v(&not(&a).unwrap()),
            vec![Some(false), Some(true), None, Some(false)]
        );
        assert!(and(&a, &BooleanArray::from_options([Some(true)])).is_err());
    }

    #[test]
    fn sliced_inputs() {
        let a = BooleanArray::from_options((0..100).map(|i| (i % 5 != 0).then_some(i % 2 == 0)));
        let b = BooleanArray::from_options((0..100).map(|i| (i % 7 != 0).then_some(i % 3 == 0)));
        let (sa, sb) = (a.slice(13, 70).unwrap(), b.slice(13, 70).unwrap());
        let r = and(&sa, &sb).unwrap();
        let want: Vec<_> = sa
            .iter()
            .zip(sb.iter())
            .map(|(x, y)| x.zip(y).map(|(x, y)| x && y))
            .collect();
        assert_eq!(v(&r), want);
        let n = not(&sa).unwrap();
        assert_eq!(v(&n), sa.iter().map(|x| x.map(|x| !x)).collect::<Vec<_>>());
    }
}
