//! sort_by and join vs naive references.
use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray};
use tessera_core::{DataFrame, DataType, Scalar, Series};

fn frame(k: &[Option<i64>], v: &[Option<i64>], vname: &str) -> DataFrame {
    DataFrame::new(vec![
        Series::new(
            "k",
            Array::Int64(PrimitiveArray::from_options(k.iter().copied())),
        ),
        Series::new(
            vname,
            Array::Int64(PrimitiveArray::from_options(v.iter().copied())),
        ),
    ])
    .unwrap()
}

fn s(o: Option<i64>) -> Scalar {
    o.map_or(Scalar::Null(DataType::Int64), Scalar::Int64)
}

proptest! {
    #[test]
    fn sort_matches_naive(k in prop::collection::vec(prop::option::of(-5i64..5), 0..100), desc in any::<bool>()) {
        let v: Vec<Option<i64>> = (0..k.len() as i64).map(Some).collect();
        let df = frame(&k, &v, "v");
        let out = df.sort_by("k", desc).unwrap();
        // Naive: stable sort of (key, original position) rows with nulls last.
        let mut rows: Vec<(Option<i64>, Option<i64>)> = k.iter().copied().zip(v.iter().copied()).collect();
        rows.sort_by(|a, b| match (a.0, b.0) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, _) => std::cmp::Ordering::Greater,
            (_, None) => std::cmp::Ordering::Less,
            (Some(x), Some(y)) => if desc { y.cmp(&x) } else { x.cmp(&y) },
        });
        for (i, r) in rows.iter().enumerate() {
            prop_assert_eq!(out.column("k").unwrap().get(i), s(r.0));
            prop_assert_eq!(out.column("v").unwrap().get(i), s(r.1));
        }
    }

    #[test]
    fn join_matches_nested_loop(lk in prop::collection::vec(prop::option::of(0i64..6), 0..40),
                                rk in prop::collection::vec(prop::option::of(0i64..6), 0..40)) {
        let lv: Vec<Option<i64>> = (0..lk.len() as i64).map(Some).collect();
        let rv: Vec<Option<i64>> = (100..100 + rk.len() as i64).map(Some).collect();
        let out = frame(&lk, &lv, "v").join(&frame(&rk, &rv, "v"), "k").unwrap();
        let mut want = Vec::new();
        for (l, a) in lk.iter().enumerate() {
            for (r, b) in rk.iter().enumerate() {
                if a.is_some() && a == b {
                    want.push((*a, lv[l], rv[r]));
                }
            }
        }
        prop_assert_eq!(out.column_names(), vec!["k", "v", "v_right"]);
        prop_assert_eq!(out.height(), want.len());
        for (i, w) in want.iter().enumerate() {
            prop_assert_eq!(out.column("k").unwrap().get(i), s(w.0));
            prop_assert_eq!(out.column("v").unwrap().get(i), s(w.1));
            prop_assert_eq!(out.column("v_right").unwrap().get(i), s(w.2));
        }
    }
}

proptest! {
    #[test]
    fn radix_sort_matches_comparison_sort(k in prop::collection::vec(prop::option::of(prop_oneof![any::<i64>(), -3i64..3]), 0..300), desc in any::<bool>()) {
        let v: Vec<Option<i64>> = (0..k.len() as i64).map(Some).collect();
        let df = frame(&k, &v, "v");
        prop_assert_eq!(df.sort_by("k", desc).unwrap(), df.sort_by_comparison("k", desc).unwrap());
    }
}
