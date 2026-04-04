//! Casts between `Boolean`, `Int32`, `Int64`, `Float64` and `Utf8`.
//!
//! Casts are lenient: a value that cannot be represented in the target type
//! becomes null instead of failing the whole array.
//!
//! * integer narrowing out of range, float to integer for NaN/infinite/out of
//!   range values (finite floats truncate toward zero) -> null;
//! * number -> bool is `!= 0` (NaN -> null); bool -> number is `0`/`1`;
//! * anything -> Utf8 uses the Rust `Display` form (`1.0f64` -> `"1"`,
//!   booleans -> `"true"`/`"false"`);
//! * Utf8 -> number parses the trimmed text, Utf8 -> bool accepts
//!   `true`/`false` in any case; unparsable text -> null.
use tessera_arrow::{Array, BooleanArray, DataType, PrimitiveArray, Utf8Array, Utf8Builder};

use crate::error::Result;

/// Cast `a` to `to`. Casting to the same type returns a clone.
pub fn cast(a: &Array, to: DataType) -> Result<Array> {
    Ok(match (a, to) {
        (Array::Boolean(_), DataType::Boolean)
        | (Array::Int32(_), DataType::Int32)
        | (Array::Int64(_), DataType::Int64)
        | (Array::Float64(_), DataType::Float64)
        | (Array::Utf8(_), DataType::Utf8) => a.clone(),

        (Array::Boolean(x), DataType::Int32) => Array::Int32(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(i32::from)),
        )),
        (Array::Boolean(x), DataType::Int64) => Array::Int64(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(i64::from)),
        )),
        (Array::Boolean(x), DataType::Float64) => Array::Float64(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(f64::from)),
        )),
        (Array::Boolean(x), DataType::Utf8) => {
            Array::Utf8(to_utf8(x.iter().map(|v| v.map(|b| b.to_string())))?)
        }

        (Array::Int32(x), DataType::Boolean) => {
            Array::Boolean(to_bool(x.iter().map(|v| v.map(|v| v != 0))))
        }
        (Array::Int32(x), DataType::Int64) => Array::Int64(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(i64::from)),
        )),
        (Array::Int32(x), DataType::Float64) => Array::Float64(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(f64::from)),
        )),
        (Array::Int32(x), DataType::Utf8) => {
            Array::Utf8(to_utf8(x.iter().map(|v| v.map(|v| v.to_string())))?)
        }

        (Array::Int64(x), DataType::Boolean) => {
            Array::Boolean(to_bool(x.iter().map(|v| v.map(|v| v != 0))))
        }
        (Array::Int64(x), DataType::Int32) => Array::Int32(PrimitiveArray::from_options(
            x.iter().map(|v| v.and_then(|v| i32::try_from(v).ok())),
        )),
        (Array::Int64(x), DataType::Float64) => Array::Float64(PrimitiveArray::from_options(
            x.iter().map(|v| v.map(|v| v as f64)),
        )),
        (Array::Int64(x), DataType::Utf8) => {
            Array::Utf8(to_utf8(x.iter().map(|v| v.map(|v| v.to_string())))?)
        }

        (Array::Float64(x), DataType::Boolean) => Array::Boolean(to_bool(
            x.iter()
                .map(|v| v.and_then(|v| (!v.is_nan()).then_some(v != 0.0))),
        )),
        (Array::Float64(x), DataType::Int32) => {
            Array::Int32(PrimitiveArray::from_options(x.iter().map(|v| {
                v.and_then(|v| {
                    let t = v.trunc();
                    (t >= f64::from(i32::MIN) && t <= f64::from(i32::MAX)).then_some(t as i32)
                })
            })))
        }
        (Array::Float64(x), DataType::Int64) => {
            Array::Int64(PrimitiveArray::from_options(x.iter().map(|v| {
                v.and_then(|v| {
                    let t = v.trunc();
                    // 2^63 is exactly representable; i64::MAX itself is not.
                    (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
                        .contains(&t)
                        .then_some(t as i64)
                })
            })))
        }
        (Array::Float64(x), DataType::Utf8) => {
            Array::Utf8(to_utf8(x.iter().map(|v| v.map(|v| v.to_string())))?)
        }

        (Array::Utf8(x), DataType::Boolean) => Array::Boolean(to_bool(x.iter().map(|v| {
            v.and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            })
        }))),
        (Array::Utf8(x), DataType::Int32) => Array::Int32(PrimitiveArray::from_options(parse(x))),
        (Array::Utf8(x), DataType::Int64) => Array::Int64(PrimitiveArray::from_options(parse(x))),
        (Array::Utf8(x), DataType::Float64) => {
            Array::Float64(PrimitiveArray::from_options(parse(x)))
        }
    })
}

fn parse<T: std::str::FromStr>(a: &Utf8Array) -> impl Iterator<Item = Option<T>> + '_ {
    a.iter().map(|v| v.and_then(|s| s.trim().parse().ok()))
}

fn to_bool(it: impl Iterator<Item = Option<bool>>) -> BooleanArray {
    BooleanArray::from_options(it)
}

fn to_utf8(it: impl Iterator<Item = Option<String>>) -> Result<Utf8Array> {
    let mut b = Utf8Builder::new();
    for v in it {
        b.push(v.as_deref());
    }
    Ok(b.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(a: &Array) -> Vec<Option<i64>> {
        match a {
            Array::Int64(x) => x.iter().collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn numeric_casts() {
        let a = Array::Int64(PrimitiveArray::from_options([
            Some(1i64),
            None,
            Some(i64::from(i32::MAX) + 1),
            Some(-7),
        ]));
        let Array::Int32(n) = cast(&a, DataType::Int32).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            n.iter().collect::<Vec<_>>(),
            vec![Some(1), None, None, Some(-7)]
        );
        let f = Array::Float64(PrimitiveArray::from_options([
            Some(2.9),
            Some(-2.9),
            Some(f64::NAN),
            Some(1e30),
            None,
            Some(9.3e18),
        ]));
        assert_eq!(
            ints(&cast(&f, DataType::Int64).unwrap()),
            vec![Some(2), Some(-2), None, None, None, None]
        );
        let Array::Boolean(b) = cast(&f, DataType::Boolean).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![Some(true), Some(true), None, Some(true), None, Some(true)]
        );
    }

    #[test]
    fn bool_and_utf8_casts_on_sliced_input() {
        let b = Array::Boolean(BooleanArray::from_options([
            Some(true),
            None,
            Some(false),
            Some(true),
        ]))
        .slice(1, 3)
        .unwrap();
        assert_eq!(
            ints(&cast(&b, DataType::Int64).unwrap()),
            vec![None, Some(0), Some(1)]
        );
        let Array::Utf8(s) = cast(&b, DataType::Utf8).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![None, Some("false"), Some("true")]
        );
        let t = Array::Utf8(
            Utf8Array::from_options([Some(" 42 "), Some("x"), None, Some("-3"), Some("TRUE")])
                .unwrap(),
        )
        .slice(1, 4)
        .unwrap();
        assert_eq!(
            ints(&cast(&t, DataType::Int64).unwrap()),
            vec![None, None, Some(-3), None]
        );
        let Array::Boolean(tb) = cast(&t, DataType::Boolean).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            tb.iter().collect::<Vec<_>>(),
            vec![None, None, None, Some(true)]
        );
        let Array::Float64(tf) = cast(&t, DataType::Float64).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            tf.iter().collect::<Vec<_>>(),
            vec![None, None, Some(-3.0), None]
        );
    }

    #[test]
    fn identity_and_float_to_utf8() {
        let f = Array::Float64(PrimitiveArray::from_options([Some(1.5), None, Some(2.0)]));
        assert_eq!(cast(&f, DataType::Float64).unwrap(), f);
        let Array::Utf8(s) = cast(&f, DataType::Utf8).unwrap() else {
            panic!("unexpected type")
        };
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![Some("1.5"), None, Some("2")]
        );
    }
}
