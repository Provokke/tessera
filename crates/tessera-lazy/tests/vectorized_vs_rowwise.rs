//! The kernel-based evaluator must equal row-wise evaluation for every
//! operator it claims to cover: column-vs-literal and column-vs-column
//! comparisons (Int32/Int64/Float64/Utf8, mixed int/float), `+ - * /`, NOT, and
//! `AND`/`OR` over bitmaps.
use proptest::prelude::*;
use tessera_arrow::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use tessera_core::{DataFrame, Scalar, Series};
use tessera_lazy::{col, lit_f64, lit_i64, lit_str, Expr};

type Row = (
    Option<i64>,
    Option<f64>,
    Option<i32>,
    Option<String>,
    Option<bool>,
);

fn rows() -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec(
        (
            prop::option::of(-10i64..10),
            prop::option::of(-10.0f64..10.0),
            prop::option::of(-10i32..10),
            prop::option::of("[abc]{0,2}"),
            prop::option::of(any::<bool>()),
        ),
        0..200,
    )
}

fn frame(rows: &[Row]) -> DataFrame {
    DataFrame::new(vec![
        Series::new(
            "i",
            Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0))),
        ),
        Series::new(
            "f",
            Array::Float64(PrimitiveArray::from_options(
                rows.iter().map(|r| r.1.map(|x| x.round() / 2.0)),
            )),
        ),
        Series::new(
            "j",
            Array::Int32(PrimitiveArray::from_options(rows.iter().map(|r| r.2))),
        ),
        Series::new(
            "s",
            Array::Utf8(Utf8Array::from_options(rows.iter().map(|r| r.3.as_deref())).unwrap()),
        ),
        Series::new(
            "b",
            Array::Boolean(BooleanArray::from_options(rows.iter().map(|r| r.4))),
        ),
    ])
    .unwrap()
}

fn num_lit() -> impl Strategy<Value = Expr> {
    prop_oneof![
        (-6i64..6).prop_map(lit_i64),
        (-12i64..12).prop_map(|k| lit_f64(k as f64 / 2.0)),
    ]
}

/// Numeric expression that always contains a column on its left spine.
fn num_expr() -> impl Strategy<Value = Expr> {
    let leaf = prop::sample::select(vec!["i", "f", "j"]).prop_map(col);
    leaf.prop_recursive(3, 12, 2, |inner| {
        (
            inner.clone(),
            prop_oneof![inner.clone(), num_lit()],
            0u8..4,
            any::<bool>(),
        )
            .prop_map(|(x, y, op, flip)| {
                let (a, b) = if flip && matches!(y, Expr::Lit(_)) {
                    (y, x)
                } else {
                    (x, y)
                };
                match op {
                    0 => a + b,
                    1 => a - b,
                    2 => a * b,
                    _ => a / b,
                }
            })
    })
}

fn cmp(a: Expr, b: Expr, op: u8) -> Expr {
    match op {
        0 => a.gt(b),
        1 => a.lt(b),
        2 => a.equals(b),
        3 => a.ge(b),
        4 => a.le(b),
        _ => a.not_equals(b),
    }
}

fn cmp_expr() -> impl Strategy<Value = Expr> {
    prop_oneof![
        (
            num_expr(),
            prop_oneof![num_expr(), num_lit()],
            0u8..6,
            any::<bool>()
        )
            .prop_map(|(a, b, op, flip)| if flip && matches!(b, Expr::Lit(_)) {
                cmp(b, a, op)
            } else {
                cmp(a, b, op)
            }),
        (
            prop_oneof![Just(col("s")), "[abc]{0,2}".prop_map(|s| lit_str(&s))],
            0u8..6
        )
            .prop_map(|(rhs, op)| cmp(col("s"), rhs, op)),
        ("[abc]{0,2}", 0u8..6).prop_map(|(s, op)| cmp(lit_str(&s), col("s"), op)),
    ]
}

fn bool_expr() -> impl Strategy<Value = Expr> {
    let leaf = prop_oneof![cmp_expr(), Just(col("b"))];
    leaf.prop_recursive(3, 10, 2, |inner| {
        (inner.clone(), inner, 0u8..4).prop_map(|(x, y, k)| match k {
            0 => x.and(y),
            1 => x.or(y),
            2 => x.not(),
            _ => x.not().and(y),
        })
    })
}

fn assert_same(e: &Expr, df: &DataFrame) -> Result<(), TestCaseError> {
    let fast = e.evaluate_vectorized(df).unwrap();
    prop_assert!(fast.is_some(), "not vectorized: {e}");
    let fast = fast.unwrap();
    let slow = e.evaluate_rowwise(df).unwrap();
    prop_assert_eq!(fast.len(), slow.len());
    prop_assert_eq!(fast.data_type(), slow.data_type());
    for i in 0..fast.len() {
        let (p, q) = (fast.scalar(i), slow.scalar(i));
        // 0.0 / 0.0 is NaN, which never equals itself under `PartialEq`.
        let both_nan = matches!((&p, &q), (Scalar::Float64(a), Scalar::Float64(b)) if a.is_nan() && b.is_nan());
        prop_assert!(both_nan || p == q, "row {} of {}: {:?} vs {:?}", i, e, p, q);
    }
    Ok(())
}

proptest! {
    #[test]
    fn fast_path_matches_rowwise(
        rows in rows(),
        k in -10i64..10, op in 0u8..6, use_float in any::<bool>()) {
        let df = frame(&rows);
        let c = col(if use_float { "f" } else { "i" });
        let e = cmp(c, Expr::Lit(Scalar::Int64(k)), op);
        assert_same(&e, &df)?;
        if let (Some(Array::Boolean(fm)), Array::Boolean(sm)) =
            (e.evaluate_vectorized(&df).unwrap(), e.evaluate_rowwise(&df).unwrap())
        {
            prop_assert_eq!(df.filter(&fm).unwrap(), df.filter(&sm).unwrap());
        } else {
            prop_assert!(false, "predicate not boolean");
        }
    }

    #[test]
    fn division_by_zero_is_null_for_ints(rows in rows(), k in -2i64..3, use_i32 in any::<bool>()) {
        let df = frame(&rows);
        let c = col(if use_i32 { "j" } else { "i" });
        let e = c / Expr::Lit(Scalar::Int64(k));
        assert_same(&e, &df)?;
        let out = e.evaluate_vectorized(&df).unwrap().unwrap();
        if k == 0 {
            for i in 0..out.len() {
                prop_assert!(out.scalar(i).is_null());
            }
        }
        // column / column: the divisor column includes zeros and nulls
        assert_same(&(col("i") / col("j")), &df)?;
    }

    #[test]
    fn numeric_exprs_match_rowwise(rows in rows(), e in num_expr()) {
        let df = frame(&rows);
        // A bare column is not evaluated by the kernels (it is just the column).
        if matches!(e, Expr::Col(_)) { return Ok(()); }
        assert_same(&e, &df)?;
    }

    #[test]
    fn comparisons_match_rowwise(rows in rows(), e in cmp_expr()) {
        assert_same(&e, &frame(&rows))?;
    }

    #[test]
    fn boolean_trees_match_rowwise(rows in rows(), e in bool_expr()) {
        let df = frame(&rows);
        if matches!(e, Expr::Col(_)) { return Ok(()); }
        assert_same(&e, &df)?;
    }
}

#[test]
fn not_division_and_comparison_examples() {
    let rows: Vec<Row> = vec![
        (Some(7), Some(1.0), Some(0), None, Some(true)),
        (Some(-7), Some(0.0), Some(2), Some("a".into()), Some(false)),
        (None, None, None, None, None),
    ];
    let df = frame(&rows);
    let get = |e: Expr| -> Vec<Scalar> {
        let a = e.evaluate(&df).unwrap();
        (0..a.len()).map(|i| a.scalar(i)).collect()
    };
    let null = |d| Scalar::Null(d);
    use tessera_core::DataType::{Boolean, Float64, Int64};
    // integer division truncates toward zero; / 0 is null (also column / column)
    assert_eq!(
        get(col("i") / lit_i64(2)),
        vec![Scalar::Int64(3), Scalar::Int64(-3), null(Int64)]
    );
    assert_eq!(
        get(col("i") / col("j")),
        vec![null(Int64), Scalar::Int64(-3), null(Int64)]
    );
    // float division is IEEE: x / 0.0 is infinite, not null
    assert_eq!(
        get(lit_i64(1) / col("f")),
        vec![
            Scalar::Float64(2.0),
            Scalar::Float64(f64::INFINITY),
            null(Float64)
        ]
    );
    assert_eq!(
        get(col("b").not()),
        vec![Scalar::Boolean(false), Scalar::Boolean(true), null(Boolean)]
    );
    assert_eq!(
        get(col("i").ge(lit_i64(7))),
        vec![Scalar::Boolean(true), Scalar::Boolean(false), null(Boolean)]
    );
    assert_eq!(
        get(col("i").le(lit_i64(-7))),
        vec![Scalar::Boolean(false), Scalar::Boolean(true), null(Boolean)]
    );
    assert_eq!(
        get(col("i").not_equals(lit_i64(7))),
        vec![Scalar::Boolean(false), Scalar::Boolean(true), null(Boolean)]
    );
    // NOT of a non-boolean is an error on both evaluation paths
    assert!(col("i").not().evaluate(&df).is_err());
    assert!(col("i").not().evaluate_rowwise(&df).is_err());
}
