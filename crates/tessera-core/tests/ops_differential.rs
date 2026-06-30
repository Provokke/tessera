//! Parallel partitioned group-by, multi-key sort, multi-column / left join and
//! `with_column` against simple row-wise references.
use proptest::prelude::*;
use tessera_arrow::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use tessera_core::{AggKind, DataFrame, DataType, JoinType, Scalar, Series};

type Row = (
    Option<i64>,
    Option<String>,
    Option<f64>,
    Option<bool>,
    Option<i32>,
);

fn row_strategy() -> impl Strategy<Value = Row> {
    (
        prop::option::of(0i64..4),
        prop::option::of("[ab]"),
        prop::option::of(prop_oneof![
            Just(0.0f64),
            Just(-0.0f64),
            Just(f64::NAN),
            Just(1.5f64),
            Just(-2.0f64)
        ]),
        prop::option::of(any::<bool>()),
        prop::option::of(-3i32..3),
    )
}

fn frame(rows: &[Row]) -> DataFrame {
    DataFrame::new(vec![
        Series::new(
            "i",
            Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0))),
        ),
        Series::new(
            "s",
            Array::Utf8(Utf8Array::from_options(rows.iter().map(|r| r.1.as_deref())).unwrap()),
        ),
        Series::new(
            "f",
            Array::Float64(PrimitiveArray::from_options(rows.iter().map(|r| r.2))),
        ),
        Series::new(
            "b",
            Array::Boolean(BooleanArray::from_options(rows.iter().map(|r| r.3))),
        ),
        Series::new(
            "n",
            Array::Int32(PrimitiveArray::from_options(rows.iter().map(|r| r.4))),
        ),
        Series::new(
            "pos",
            Array::Int64(PrimitiveArray::from_options(
                (0..rows.len() as i64).map(Some),
            )),
        ),
    ])
    .unwrap()
}

/// Frame equality that treats NaN == NaN.
fn same(a: &DataFrame, b: &DataFrame) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.column_names(), b.column_names());
    prop_assert_eq!(a.height(), b.height());
    for c in a.column_names() {
        let (x, y) = (a.column(c).unwrap(), b.column(c).unwrap());
        prop_assert_eq!(x.data_type(), y.data_type());
        for i in 0..a.height() {
            let (p, q) = (x.get(i), y.get(i));
            let nan = matches!((&p, &q), (Scalar::Float64(u), Scalar::Float64(v)) if u.is_nan() && v.is_nan());
            prop_assert!(nan || p == q, "col {} row {}: {:?} vs {:?}", c, i, p, q);
        }
    }
    Ok(())
}

proptest! {
    #[test]
    fn partitioned_group_by_matches_reference(rows in prop::collection::vec(row_strategy(), 0..150),
                                              keys in prop::sample::subsequence(vec!["i", "s", "f", "b", "n"], 1..=4),
                                              parts in 1usize..9) {
        let df = frame(&rows);
        let aggs = [("pos", AggKind::Sum), ("pos", AggKind::Min), ("pos", AggKind::Max),
                    ("pos", AggKind::Count), ("pos", AggKind::Mean), ("f", AggKind::Sum), ("f", AggKind::Max)];
        let want = df.group_by_reference(&keys, &aggs).unwrap();
        let got = df.group_by_with_partitions(&keys, &aggs, parts).unwrap();
        same(&want, &got)?;
        same(&want, &df.group_by(&keys, &aggs).unwrap())?;
    }

    #[test]
    fn multi_key_sort_matches_comparison(rows in prop::collection::vec(row_strategy(), 0..120),
                                         keys in prop::sample::subsequence(vec!["i", "s", "f", "b", "n"], 1..=5),
                                         desc in prop::collection::vec(any::<bool>(), 5)) {
        let df = frame(&rows);
        let d = &desc[..keys.len()];
        let want = df.sort_by_multi_comparison(&keys, d).unwrap();
        let got = df.sort_by_multi(&keys, d).unwrap();
        same(&want, &got)?;
    }

    #[test]
    fn multi_column_join_matches_nested_loop(
        l in prop::collection::vec((prop::option::of(0i64..3), prop::option::of("[ab]")), 0..30),
        r in prop::collection::vec((prop::option::of(0i64..3), prop::option::of("[ab]")), 0..30),
        left in any::<bool>()) {
        let mk = |rows: &[(Option<i64>, Option<String>)], v: &str, base: i64| DataFrame::new(vec![
            Series::new("k1", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|x| x.0)))),
            Series::new("k2", Array::Utf8(Utf8Array::from_options(rows.iter().map(|x| x.1.as_deref())).unwrap())),
            Series::new(v, Array::Int64(PrimitiveArray::from_options((0..rows.len() as i64).map(|i| Some(base + i))))),
        ]).unwrap();
        let (a, b) = (mk(&l, "v", 0), mk(&r, "v", 100));
        let how = if left { JoinType::Left } else { JoinType::Inner };
        let out = a.join_on(&b, &["k1", "k2"], how).unwrap();
        // (left row, Some(right row) | None)
        let mut want: Vec<(usize, Option<usize>)> = Vec::new();
        for (i, x) in l.iter().enumerate() {
            let mut hit = false;
            for (j, y) in r.iter().enumerate() {
                if x.0.is_some() && x.1.is_some() && x == y {
                    want.push((i, Some(j)));
                    hit = true;
                }
            }
            if !hit && left {
                want.push((i, None));
            }
        }
        prop_assert_eq!(out.column_names(), vec!["k1", "k2", "v", "v_right"]);
        prop_assert_eq!(out.height(), want.len());
        let v = |o: Option<i64>| o.map_or(Scalar::Null(DataType::Int64), Scalar::Int64);
        for (n, (i, j)) in want.iter().enumerate() {
            prop_assert_eq!(out.column("v").unwrap().get(n), v(Some(*i as i64)));
            prop_assert_eq!(out.column("v_right").unwrap().get(n), v(j.map(|j| 100 + j as i64)));
            prop_assert_eq!(out.column("k1").unwrap().get(n), v(l[*i].0));
        }
    }
}

#[test]
fn large_default_group_by_uses_many_partitions_and_matches_reference() {
    let rows: Vec<Row> = (0..120_000u64)
        .map(|i| {
            let h = i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 33;
            (
                (h % 11 != 0).then_some((h % 300) as i64),
                (h % 13 != 0).then(|| format!("s{}", h % 7)),
                Some((h % 5) as f64 / 2.0),
                None,
                Some((h % 3) as i32),
            )
        })
        .collect();
    let df = frame(&rows);
    let aggs = [
        ("pos", AggKind::Sum),
        ("pos", AggKind::Mean),
        ("f", AggKind::Min),
        ("pos", AggKind::Count),
    ];
    for keys in [vec!["i", "s"], vec!["s", "n", "f"], vec!["s"]] {
        let want = df.group_by_reference(&keys, &aggs).unwrap();
        let got = df.group_by(&keys, &aggs).unwrap();
        assert!(same(&want, &got).is_ok(), "keys {keys:?}");
        assert!(want.height() > 1);
    }
}

#[test]
fn with_column_adds_replaces_and_validates() {
    let df = frame(&[
        (Some(1), None, None, None, None),
        (Some(2), None, None, None, None),
    ]);
    let extra = Series::new(
        "x",
        Array::Int64(PrimitiveArray::from_options([Some(7), None])),
    );
    let added = df.with_column(extra.clone()).unwrap();
    assert_eq!(added.shape(), (2, 7));
    assert_eq!(added.column_names().last(), Some(&"x"));
    // Replace keeps position and may change the type.
    let repl = Series::new(
        "i",
        Array::Utf8(Utf8Array::from_options([Some("a"), Some("b")]).unwrap()),
    );
    let r = df.with_column(repl).unwrap();
    assert_eq!(r.column_names(), df.column_names());
    assert_eq!(r.column("i").unwrap().get(1), Scalar::Utf8("b".into()));
    // Wrong length is rejected; an empty frame accepts any length.
    let short = Series::new("y", Array::Int64(PrimitiveArray::from_options([Some(1)])));
    assert!(df.with_column(short.clone()).is_err());
    assert_eq!(
        DataFrame::default().with_column(short).unwrap().shape(),
        (1, 1)
    );
}

#[test]
fn left_join_keeps_unmatched_rows_with_nulls() {
    let a = DataFrame::new(vec![
        Series::new(
            "k",
            Array::Int64(PrimitiveArray::from_options([Some(1), Some(2), None])),
        ),
        Series::new(
            "a",
            Array::Int64(PrimitiveArray::from_options([Some(10), Some(20), Some(30)])),
        ),
    ])
    .unwrap();
    let b = DataFrame::new(vec![
        Series::new(
            "k",
            Array::Int32(PrimitiveArray::from_options([Some(2), Some(2)])),
        ),
        Series::new(
            "b",
            Array::Utf8(Utf8Array::from_options([Some("x"), Some("y")]).unwrap()),
        ),
    ])
    .unwrap();
    let out = a.join_on(&b, &["k"], JoinType::Left).unwrap();
    assert_eq!(out.height(), 4);
    let bs: Vec<Scalar> = (0..4).map(|i| out.column("b").unwrap().get(i)).collect();
    assert_eq!(
        bs,
        vec![
            Scalar::Null(DataType::Utf8),
            Scalar::Utf8("x".into()),
            Scalar::Utf8("y".into()),
            Scalar::Null(DataType::Utf8)
        ]
    );
    // Inner join drops the unmatched and the null-key row.
    assert_eq!(a.join_on(&b, &["k"], JoinType::Inner).unwrap().height(), 2);
}
