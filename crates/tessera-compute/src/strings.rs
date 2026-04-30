//! String kernels over [`Utf8Array`]. Null in, null out.
use memchr::memmem;
use tessera_arrow::{Bitmap, BooleanArray, PrimitiveArray, Utf8Array, Utf8Builder};

use crate::error::Result;

/// Bitmap of rows whose value contains `pattern` (`memchr::memmem`). An empty
/// pattern matches every non-null row. Null rows are null in the result.
pub fn contains(a: &Utf8Array, pattern: &str) -> Result<BooleanArray> {
    let finder = memmem::Finder::new(pattern.as_bytes());
    let bits = Bitmap::from_bools((0..a.len()).map(|i| {
        a.get(i)
            .is_some_and(|s| finder.find(s.as_bytes()).is_some())
    }));
    Ok(BooleanArray::try_new(bits, a.validity().cloned())?)
}

/// Length in Unicode scalar values (`char`s) as `i32`.
pub fn char_len(a: &Utf8Array) -> Result<PrimitiveArray<i32>> {
    let values = (0..a.len())
        .map(|i| {
            let s = a.value(i);
            // ASCII fast path: the byte length is the char length.
            let n = if s.is_ascii() {
                s.len()
            } else {
                s.chars().count()
            };
            n as i32
        })
        .collect();
    Ok(PrimitiveArray::try_new(values, a.validity().cloned())?)
}

/// Upper-case every string (full Unicode mapping, so lengths may change).
pub fn upper(a: &Utf8Array) -> Result<Utf8Array> {
    let mut b = Utf8Builder::new();
    for i in 0..a.len() {
        match a.get(i) {
            Some(s) if s.is_ascii() => b.push(Some(&s.to_ascii_uppercase())),
            Some(s) => b.push(Some(&s.to_uppercase())),
            None => b.push(None),
        }
    }
    Ok(b.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Utf8Array {
        Utf8Array::from_options([
            Some("hello world"),
            None,
            Some(""),
            Some("stra\u{df}e"),
            Some("h\u{e9}llo"),
            Some("world"),
        ])
        .unwrap()
    }

    #[test]
    fn contains_with_nulls_and_slices() {
        let a = sample();
        let r = contains(&a, "world").unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(true)
            ]
        );
        let r = contains(&a.slice(1, 4).unwrap(), "").unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![None, Some(true), Some(true), Some(true)]
        );
        let r = contains(&a.slice(3, 3).unwrap(), "llo").unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(false), Some(true), Some(false)]
        );
    }

    #[test]
    fn char_len_counts_chars_not_bytes() {
        let a = sample();
        let r = char_len(&a).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(11), None, Some(0), Some(6), Some(5), Some(5)]
        );
        let r = char_len(&a.slice(3, 2).unwrap()).unwrap();
        assert_eq!(r.iter().collect::<Vec<_>>(), vec![Some(6), Some(5)]);
    }

    #[test]
    fn upper_handles_unicode_and_nulls() {
        let a = sample();
        let r = upper(&a).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![
                Some("HELLO WORLD"),
                None,
                Some(""),
                Some("STRASSE"),
                Some("H\u{c9}LLO"),
                Some("WORLD")
            ]
        );
        let r = upper(&a.slice(2, 3).unwrap()).unwrap();
        assert_eq!(
            r.iter().collect::<Vec<_>>(),
            vec![Some(""), Some("STRASSE"), Some("H\u{c9}LLO")]
        );
    }
}
