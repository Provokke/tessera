//! Right and full outer joins, and differently named join keys, against a
//! nested-loop reference (exact row order included).
use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray};
use tessera_core::{DataFrame, DataType, JoinType, Scalar, Series};

type Cell = Option<i64>;
type Row = (Cell, Cell, Cell);

fn rows() -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec(
        (
            prop::option::of(0i64..4),
            prop::option::of(0i64..3),
            prop::option::of(-9i64..9),
        ),
        0..30,
    )
}

fn frame(names: [&str; 3], rows: &[Row]) -> DataFrame {
    let col = |f: fn(&Row) -> Cell| Array::Int64(PrimitiveArray::from_options(rows.iter().map(f)));
    DataFrame::new(vec![
        Series::new(names[0], col(|r| r.0)),
        Series::new(names[1], col(|r| r.1)),
        Series::new(names[2], col(|r| r.2)),
    ])
    .unwrap()
}

fn s(o: Cell) -> Scalar {
    o.map_or(Scalar::Null(DataType::Int64), Scalar::Int64)
}

/// Row pairs in the documented order: probe results in left order (unmatched
/// left rows kept for Left/Outer), then unmatched right rows for Right/Outer.
fn reference(
    l: &[Row],
    r: &[Row],
    how: JoinType,
    keys: usize,
) -> Vec<(Option<usize>, Option<usize>)> {
    let key = |x: &Row| -> Option<Vec<i64>> {
        let all = [x.0, x.1];
        all[..keys].iter().copied().collect()
    };
    let mut out = Vec::new();
    let mut right_seen = vec![false; r.len()];
    for (i, a) in l.iter().enumerate() {
        let mut matched = false;
        for (j, b) in r.iter().enumerate() {
            if let (Some(x), Some(y)) = (key(a), key(b)) {
                if x == y {
                    out.push((Some(i), Some(j)));
                    right_seen[j] = true;
                    matched = true;
                }
            }
        }
        if !matched && matches!(how, JoinType::Left | JoinType::Outer) {
            out.push((Some(i), None));
        }
    }
    if matches!(how, JoinType::Right | JoinType::Outer) {
        for (j, seen) in right_seen.iter().enumerate() {
            if !seen {
                out.push((None, Some(j)));
            }
        }
    }
    out
}

fn how_of(n: u8) -> JoinType {
    [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Outer,
    ][n as usize % 4]
}

proptest! {
    /// Different key names on each side: both key columns are kept and stay
    /// null on the side that has no row.
    #[test]
    fn differently_named_keys_match_nested_loop(
        l in rows(), r in rows(), h in 0u8..4, two in any::<bool>()) {
        let how = how_of(h);
        let (left, right) = (frame(["a", "b", "lv"], &l), frame(["c", "d", "rv"], &r));
        let (lo, ro): (&[&str], &[&str]) = if two { (&["a", "b"], &["c", "d"]) } else { (&["a"], &["c"]) };
        let out = left.join_keys(&right, lo, ro, how).unwrap();
        let want = reference(&l, &r, how, lo.len());
        prop_assert_eq!(out.column_names(), vec!["a", "b", "lv", "c", "d", "rv"]);
        prop_assert_eq!(out.height(), want.len());
        for (k, (i, j)) in want.iter().enumerate() {
            let lc = |f: fn(&Row) -> Cell| s(i.and_then(|i| f(&l[i])));
            let rc = |f: fn(&Row) -> Cell| s(j.and_then(|j| f(&r[j])));
            prop_assert_eq!(out.column("a").unwrap().get(k), lc(|x| x.0));
            prop_assert_eq!(out.column("b").unwrap().get(k), lc(|x| x.1));
            prop_assert_eq!(out.column("lv").unwrap().get(k), lc(|x| x.2));
            prop_assert_eq!(out.column("c").unwrap().get(k), rc(|x| x.0));
            prop_assert_eq!(out.column("d").unwrap().get(k), rc(|x| x.1));
            prop_assert_eq!(out.column("rv").unwrap().get(k), rc(|x| x.2));
        }
    }

    /// Same-named keys: the right key column is dropped and the left key
    /// column is filled from it on right-only rows.
    #[test]
    fn same_named_keys_match_nested_loop(l in rows(), r in rows(), h in 0u8..4, two in any::<bool>()) {
        let how = how_of(h);
        let (left, right) = (frame(["a", "b", "lv"], &l), frame(["a", "b", "rv"], &r));
        let on: &[&str] = if two { &["a", "b"] } else { &["a"] };
        let out = left.join_on(&right, on, how).unwrap();
        let want = reference(&l, &r, how, on.len());
        let names = if two { vec!["a", "b", "lv", "rv"] } else { vec!["a", "b", "lv", "b_right", "rv"] };
        prop_assert_eq!(out.column_names(), names);
        prop_assert_eq!(out.height(), want.len());
        for (k, (i, j)) in want.iter().enumerate() {
            let pick = |f: fn(&Row) -> Cell| s(match (i, j) {
                (Some(i), _) => f(&l[*i]),
                (None, Some(j)) => f(&r[*j]),
                _ => None,
            });
            prop_assert_eq!(out.column("a").unwrap().get(k), pick(|x| x.0));
            if two {
                prop_assert_eq!(out.column("b").unwrap().get(k), pick(|x| x.1));
            } else {
                prop_assert_eq!(out.column("b").unwrap().get(k), s(i.and_then(|i| l[i].1)));
                prop_assert_eq!(out.column("b_right").unwrap().get(k), s(j.and_then(|j| r[j].1)));
            }
            prop_assert_eq!(out.column("lv").unwrap().get(k), s(i.and_then(|i| l[i].2)));
            prop_assert_eq!(out.column("rv").unwrap().get(k), s(j.and_then(|j| r[j].2)));
        }
    }
}

#[test]
fn right_and_outer_keep_null_key_rows_and_fill_the_key() {
    let l = frame(
        ["k", "x", "lv"],
        &[(Some(1), None, Some(10)), (None, None, Some(11))],
    );
    let r = frame(
        ["k", "x", "rv"],
        &[
            (Some(1), None, Some(20)),
            (Some(2), None, Some(21)),
            (None, None, Some(22)),
        ],
    );
    let right = l.join_keys(&r, &["k"], &["k"], JoinType::Right).unwrap();
    assert_eq!(right.height(), 3);
    assert_eq!(right.column("k").unwrap().get(0), Scalar::Int64(1));
    assert_eq!(right.column("k").unwrap().get(1), Scalar::Int64(2));
    assert_eq!(
        right.column("k").unwrap().get(2),
        Scalar::Null(DataType::Int64)
    );
    assert_eq!(
        right.column("lv").unwrap().get(1),
        Scalar::Null(DataType::Int64)
    );
    let outer = l.join_keys(&r, &["k"], &["k"], JoinType::Outer).unwrap();
    // matched (1,1), unmatched left null-key row, then right-only rows 2 and null.
    assert_eq!(outer.height(), 4);
    assert_eq!(
        outer.column("rv").unwrap().get(1),
        Scalar::Null(DataType::Int64)
    );
    assert_eq!(outer.column("rv").unwrap().get(3), Scalar::Int64(22));
}

#[test]
fn key_errors_and_widening() {
    let l = frame(["a", "b", "v"], &[(Some(1), Some(1), Some(1))]);
    assert!(l.join_keys(&l, &[], &[], JoinType::Inner).is_err());
    assert!(l
        .join_keys(&l, &["a", "b"], &["a"], JoinType::Inner)
        .is_err());
    assert!(l.join_keys(&l, &["a"], &["nope"], JoinType::Inner).is_err());
    // Same-named key of incompatible types cannot be filled in a right join.
    let strs = DataFrame::new(vec![Series::new(
        "a",
        Array::Utf8(tessera_arrow::Utf8Array::from_options([Some("x")]).unwrap()),
    )])
    .unwrap();
    assert!(l.join_keys(&strs, &["a"], &["a"], JoinType::Right).is_err());
    assert!(l.join_keys(&strs, &["a"], &["a"], JoinType::Inner).is_ok());
    // Int32 right key widens the filled left key to Int64.
    let narrow = DataFrame::new(vec![Series::new(
        "a",
        Array::Int32(PrimitiveArray::from_options([Some(1), Some(7)])),
    )])
    .unwrap();
    let out = l
        .join_keys(&narrow, &["a"], &["a"], JoinType::Outer)
        .unwrap();
    assert_eq!(out.column("a").unwrap().data_type(), DataType::Int64);
    assert_eq!(out.column("a").unwrap().get(1), Scalar::Int64(7));
    assert_eq!(
        tessera_core::merged_key_type(DataType::Int32, DataType::Int64),
        Some(DataType::Int64)
    );
}
