#![no_main]
//! Decode bytes into an expression tree and evaluate it on a small frame with
//! nulls; evaluation must never panic, the vectorized fast path must agree
//! with row-wise evaluation, and filtering by a boolean result must work.
//! The expression is also run through a join / filter / sort / group-by /
//! limit plan, which must give the same result optimised and unoptimised.
use libfuzzer_sys::fuzz_target;
use tessera_arrow::{Array, PrimitiveArray, Utf8Array};
use tessera_core::{AggKind, DataFrame, JoinType, Scalar, Series};
use tessera_lazy::{col, Expr, LogicalPlan};

fn decode(data: &mut &[u8], depth: u8) -> Expr {
    let Some((&b, rest)) = data.split_first() else { return col("i") };
    *data = rest;
    if depth > 6 || b % 4 == 0 {
        return match b % 5 {
            0 => col("i"),
            1 => col("f"),
            2 => col("s"),
            3 => Expr::Lit(Scalar::Int64((b as i64 - 128) * 1_000_000_000_000_000)),
            _ => Expr::Lit(Scalar::Float64(b as f64 / 7.0)),
        };
    }
    let l = decode(data, depth + 1);
    let r = decode(data, depth + 1);
    match (b >> 2) % 14 {
        0 | 1 => l + r,
        2 => l - r,
        3 => l * r,
        4 => l.gt(r),
        5 => l.lt(r),
        6 => l.equals(r),
        7 => l.ge(r),
        8 => l.le(r),
        9 => l.not_equals(r),
        10 => l / r,
        11 => l.not(),
        12 => l.and(r),
        _ => l.or(r),
    }
}

fuzz_target!(|data: &[u8]| {
    let df = DataFrame::new(vec![
        Series::new("i", Array::Int64(PrimitiveArray::from_options([Some(i64::MAX), None, Some(-3), Some(0)]))),
        Series::new("f", Array::Float64(PrimitiveArray::from_options([Some(f64::NAN), Some(1.5), None, Some(-0.0)]))),
        Series::new("s", Array::Utf8(Utf8Array::from_options([Some("a"), None, Some(""), Some("zz")]).unwrap())),
    ])
    .unwrap();
    let mut d = data;
    let e = decode(&mut d, 0);
    if let (Ok(Some(fast)), Ok(slow)) = (e.evaluate_vectorized(&df), e.evaluate_rowwise(&df)) {
        assert_eq!(fast.len(), slow.len());
        for i in 0..fast.len() {
            // Debug text, so NaN (e.g. 0.0 / 0.0) compares equal to itself.
            assert_eq!(format!("{:?}", fast.scalar(i)), format!("{:?}", slow.scalar(i)));
        }
    }
    let _ = e.evaluate(&df);
    let right = df.select(&["i", "s"]).unwrap();
    let plan = LogicalPlan::scan(df.clone())
        .join(LogicalPlan::scan(right), &["i"], JoinType::Inner)
        .filter(e.clone())
        .sort(&["i"], &[false])
        .group_by(&["i"], &[("f", AggKind::Count)])
        .limit(2);
    let (plain, optimized) = (plan.collect(), plan.clone().optimize().collect());
    assert_eq!(plain.is_ok(), optimized.is_ok());
    if let (Ok(a), Ok(b)) = (plain, optimized) {
        assert_eq!(a.to_string(), b.to_string());
    }
    let _ = LogicalPlan::scan(df).filter(e).optimize().collect();
});
