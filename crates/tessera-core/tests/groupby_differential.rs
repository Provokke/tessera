//! Group-by vs a naive reference (BTreeMap over rows).
use std::collections::BTreeMap;

use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray, Utf8Array};
use tessera_core::{AggKind, DataFrame, Scalar, Series};

type GroupKey = (Option<i64>, Option<String>);

proptest! {
    #[test]
    fn groupby_matches_naive(rows in prop::collection::vec(
        (prop::option::of(0i64..5), prop::option::of("[ab]"), prop::option::of(-100i64..100)), 0..200)) {
        let df = DataFrame::new(vec![
            Series::new("k1", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0)))),
            Series::new("k2", Array::Utf8(Utf8Array::from_options(rows.iter().map(|r| r.1.as_deref())).unwrap())),
            Series::new("v", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.2)))),
        ]).unwrap();
        let aggs = [("v", AggKind::Sum), ("v", AggKind::Min), ("v", AggKind::Max), ("v", AggKind::Count), ("v", AggKind::Mean)];
        let out = df.group_by(&["k1", "k2"], &aggs).unwrap();

        let mut naive: BTreeMap<GroupKey, Vec<Option<i64>>> = BTreeMap::new();
        for r in &rows {
            naive.entry((r.0, r.1.clone())).or_default().push(r.2);
        }
        prop_assert_eq!(out.height(), naive.len());
        for g in 0..out.height() {
            let k1 = match out.column("k1").unwrap().get(g) { Scalar::Int64(v) => Some(v), _ => None };
            let k2 = match out.column("k2").unwrap().get(g) { Scalar::Utf8(s) => Some(s), _ => None };
            let vals: Vec<i64> = naive[&(k1, k2)].iter().flatten().copied().collect();
            let get = |c: &str| out.column(c).unwrap().get(g);
            let opt = |o: Option<i64>| o.map_or(Scalar::Null(tessera_core::DataType::Int64), Scalar::Int64);
            prop_assert_eq!(get("v_sum"), opt((!vals.is_empty()).then(|| vals.iter().sum())));
            prop_assert_eq!(get("v_min"), opt(vals.iter().min().copied()));
            prop_assert_eq!(get("v_max"), opt(vals.iter().max().copied()));
            prop_assert_eq!(get("v_count"), Scalar::Int64(vals.len() as i64));
            let mean = (!vals.is_empty()).then(|| vals.iter().sum::<i64>() as f64 / vals.len() as f64);
            prop_assert_eq!(get("v_mean"), mean.map_or(Scalar::Null(tessera_core::DataType::Float64), Scalar::Float64));
        }
    }
}

proptest! {
    /// Single Int64 key takes the column-wise fast path; compare it with the
    /// generic multi-key path (same key column duplicated as a Utf8 copy is not
    /// needed: grouping by [k, k2] where k2 is constant yields the same groups).
    #[test]
    fn int_key_fast_path_matches_generic(rows in prop::collection::vec(
        (prop::option::of(0i64..6), prop::option::of(-100i64..100), prop::option::of(-8i32..8)), 0..200)) {
        let n = rows.len();
        let df = DataFrame::new(vec![
            Series::new("k", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0)))),
            Series::new("c", Array::Utf8(Utf8Array::from_options((0..n).map(|_| Some("x"))).unwrap())),
            Series::new("v", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.1)))),
            Series::new("f", Array::Float64(PrimitiveArray::from_options(rows.iter().map(|r| r.2.map(|x| if x == 7 { f64::NAN } else { x as f64 / 4.0 }))))),
        ]).unwrap();
        let aggs = [("v", AggKind::Sum), ("v", AggKind::Min), ("v", AggKind::Max), ("v", AggKind::Count), ("v", AggKind::Mean),
                    ("f", AggKind::Sum), ("f", AggKind::Min), ("f", AggKind::Max), ("f", AggKind::Count), ("f", AggKind::Mean)];
        let fast = df.group_by(&["k"], &aggs).unwrap();
        let generic = df.group_by(&["k", "c"], &aggs).unwrap();
        prop_assert_eq!(fast.height(), generic.height());
        for c in fast.column_names() {
            let (a, b) = (fast.column(c).unwrap(), generic.column(c).unwrap());
            for i in 0..fast.height() {
                let (x, y) = (a.get(i), b.get(i));
                let both_nan = matches!((&x, &y), (Scalar::Float64(p), Scalar::Float64(q)) if p.is_nan() && q.is_nan());
                prop_assert!(both_nan || x == y, "col {} row {}: {:?} vs {:?}", c, i, x, y);
            }
        }
    }
}
