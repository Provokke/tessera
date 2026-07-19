//! Optimised and unoptimised plans must produce identical results, over every
//! plan node, plus checks that the rules actually fire.
use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray};
use tessera_core::{AggKind, DataFrame, JoinType, Series};
use tessera_io::CsvOptions;
use tessera_lazy::{col, lit_i64, Expr, LogicalPlan};

fn pred_over(cols: Vec<&'static str>) -> impl Strategy<Value = Expr> {
    let leaf = (prop::sample::select(cols), -5i64..5, 0u8..9).prop_map(|(c, k, op)| match op {
        0 => col(c).gt(lit_i64(k)),
        1 => col(c).lt(lit_i64(k)),
        2 => lit_i64(k).lt(lit_i64(2)), // constant, folded
        3 => col(c).ge(lit_i64(k)),
        4 => col(c).le(lit_i64(k)),
        5 => col(c).not_equals(lit_i64(k)),
        6 => col(c).gt(lit_i64(k)).not(),
        7 => (col(c) / lit_i64(k)).gt(lit_i64(0)), // k == 0 gives null
        _ => (col(c) + lit_i64(k)).equals(col(c) * lit_i64(1)),
    });
    leaf.prop_recursive(2, 8, 2, |inner| {
        (inner.clone(), inner, any::<bool>())
            .prop_map(|(x, y, and)| if and { x.and(y) } else { x.or(y) })
    })
}

fn pred() -> impl Strategy<Value = Expr> {
    pred_over(vec!["a", "b"])
}

/// Logical contents (values under null slots are unspecified).
fn logical(df: &DataFrame) -> Vec<(String, Vec<tessera_core::Scalar>)> {
    df.columns()
        .iter()
        .map(|c| {
            (
                c.name().to_owned(),
                (0..c.len()).map(|i| c.get(i)).collect(),
            )
        })
        .collect()
}

type Rows = Vec<(Option<i64>, Option<i64>)>;

fn rows() -> impl Strategy<Value = Rows> {
    prop::collection::vec(
        (prop::option::of(-5i64..5), prop::option::of(-5i64..5)),
        0..60,
    )
}

fn frame(names: (&str, &str), rows: &Rows) -> DataFrame {
    DataFrame::new(vec![
        Series::new(
            names.0,
            Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0))),
        ),
        Series::new(
            names.1,
            Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.1))),
        ),
    ])
    .unwrap()
}

proptest! {
    #[test]
    fn optimize_preserves_results(
        rows in rows(),
        p1 in pred(), p2 in pred(), p3 in pred()) {
        let df = frame(("a", "b"), &rows);
        let plan = LogicalPlan::scan(df).filter(p1).filter(p2).select(&["b", "a"]).filter(p3).select(&["a", "b"]);
        let opt = plan.clone().optimize();
        prop_assert_eq!(logical(&plan.collect().unwrap()), logical(&opt.collect().unwrap()));
    }

    #[test]
    fn optimize_preserves_results_pipeline(
        left in rows(), right in rows(),
        p1 in pred(), pr in pred_over(vec!["a", "c"]), p3 in pred_over(vec!["a", "b", "s", "c"]),
        p4 in pred_over(vec!["a", "s_sum"]),
        left_join in any::<bool>(), desc in any::<bool>(), n in 0usize..8) {
        let how = if left_join { JoinType::Left } else { JoinType::Inner };
        let plan = LogicalPlan::scan(frame(("a", "b"), &left))
            .filter(p1)
            .with_columns(vec![(col("a") + col("b")).alias("s")])
            .join(LogicalPlan::scan(frame(("a", "c"), &right)).filter(pr), &["a"], how)
            .filter(p3)
            .sort(&["s", "c"], &[desc, !desc])
            .group_by(&["a"], &[("s", AggKind::Sum), ("c", AggKind::Max), ("b", AggKind::Count)])
            .filter(p4)
            .sort(&["a"], &[false])
            .limit(n)
            .select(&["a", "c_max"]);
        let opt = plan.clone().optimize();
        let (x, y) = (plan.collect(), opt.collect());
        prop_assert_eq!(x.is_ok(), y.is_ok());
        if let (Ok(x), Ok(y)) = (x, y) {
            prop_assert_eq!(logical(&x), logical(&y));
        }
    }
}

proptest! {
    /// Every join type, same- and differently-named keys: pushdown below the
    /// join must not change the result.
    #[test]
    fn optimize_preserves_results_all_joins(
        left in rows(), right in rows(),
        pl in pred_over(vec!["a", "b"]), pr in pred_over(vec!["c"]),
        pt in pred_over(vec!["a", "b", "c"]), pk in pred_over(vec!["k", "c"]),
        how in 0u8..4, renamed in any::<bool>()) {
        let how = [JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Outer][how as usize];
        let rname = if renamed { "k" } else { "a" };
        let r = LogicalPlan::scan(frame((rname, "c"), &right));
        let l = LogicalPlan::scan(frame(("a", "b"), &left));
        let joined = l.join_with(r, &["a"], &[rname], how);
        let last = if renamed { pk } else { pt.clone() };
        let plan = joined.filter(pl).filter(pr).filter(pt).filter(last);
        let opt = plan.clone().optimize();
        let (x, y) = (plan.collect(), opt.collect());
        prop_assert_eq!(x.is_ok(), y.is_ok());
        if let (Ok(x), Ok(y)) = (x, y) {
            prop_assert_eq!(logical(&x), logical(&y));
        }
    }
}

/// Position of `needle` in the plan text: the plan prints parents before
/// children, so a larger offset means "closer to the leaves".
fn at(text: &str, needle: &str) -> usize {
    text.find(needle)
        .unwrap_or_else(|| panic!("{needle:?} not in\n{text}"))
}

proptest! {
    #[test]
    fn pushdown_below_groupby_withcolumns_limit_preserves_results(
        rows in rows(),
        pk in pred_over(vec!["a"]),
        pagg in pred_over(vec!["a", "s_sum"]),
        pnew in pred_over(vec!["a", "t", "b"]),
        pany in pred_over(vec!["a", "b"]),
        pcnt in pred_over(vec!["a", "b_count"]),
        pnew2 in pred_over(vec!["a", "t", "b_sum"]),
        n in 0usize..8, which in 0u8..4) {
        let base = LogicalPlan::scan(frame(("a", "b"), &rows));
        let plan = match which {
            0 => base
                .with_columns(vec![(col("a") * lit_i64(2)).alias("t"), (col("b") - col("a")).alias("u")])
                .filter(pnew)
                .group_by(&["a"], &[("b", AggKind::Sum)])
                .filter(pk),
            1 => base
                .with_columns(vec![(col("a") + col("b")).alias("s")])
                .group_by(&["a"], &[("s", AggKind::Sum), ("b", AggKind::Count)])
                .filter(pagg)
                .filter(pcnt),
            2 => base.limit(n).filter(pany).group_by(&["b"], &[("a", AggKind::Max)]),
            _ => base
                .group_by(&["a"], &[("b", AggKind::Sum)])
                .limit(n)
                .filter(pk)
                .with_columns(vec![(col("a") + lit_i64(1)).alias("t")])
                .filter(pnew2),
        };
        let opt = plan.clone().optimize();
        let (x, y) = (plan.collect(), opt.collect());
        prop_assert_eq!(x.is_ok(), y.is_ok());
        if let (Ok(x), Ok(y)) = (x, y) {
            prop_assert_eq!(logical(&x), logical(&y));
        }
    }
}

#[test]
fn filter_moves_below_groupby_only_on_keys() {
    let df = frame(
        ("a", "b"),
        &vec![
            (Some(1), Some(10)),
            (Some(2), Some(20)),
            (Some(1), Some(5)),
            (None, Some(1)),
        ],
    );
    let plan = LogicalPlan::scan(df)
        .group_by(&["a"], &[("b", AggKind::Sum)])
        .filter(col("a").gt(lit_i64(1)).and(col("b_sum").gt(lit_i64(6))));
    let text = plan.clone().optimize().to_string();
    // the key conjunct ends up below the group-by (in the scan), the
    // aggregate conjunct stays above it
    assert!(
        at(&text, "FILTER (col(b_sum) > 6)") < at(&text, "GROUP_BY"),
        "{text}"
    );
    assert!(text.contains("filter=(col(a) > 1)"), "{text}");
    assert!(!text.contains("FILTER (col(a)"), "{text}");
    assert_eq!(
        logical(&plan.collect().unwrap()),
        logical(&plan.optimize().collect().unwrap())
    );
}

#[test]
fn filter_on_float_key_with_division_stays_above_groupby() {
    // -0.0 and 0.0 are one group (key 0.0, first row), but 1 / -0.0 < 0.
    let f = DataFrame::new(vec![
        Series::new(
            "k",
            Array::Float64(PrimitiveArray::from_options([
                Some(0.0),
                Some(-0.0),
                Some(2.0),
            ])),
        ),
        Series::new(
            "v",
            Array::Int64(PrimitiveArray::from_options([Some(1), Some(2), Some(3)])),
        ),
    ])
    .unwrap();
    let plan = LogicalPlan::scan(f.clone())
        .group_by(&["k"], &[("v", AggKind::Sum)])
        .filter((lit_i64(1) / col("k")).gt(lit_i64(0)));
    let opt = plan.clone().optimize();
    let text = opt.to_string();
    assert!(at(&text, "FILTER") < at(&text, "GROUP_BY"), "{text}");
    let (x, y) = (plan.collect().unwrap(), opt.collect().unwrap());
    assert_eq!(x.shape(), (2, 2));
    assert_eq!(logical(&x), logical(&y));
    // without a division the same float key does move below
    let plan = LogicalPlan::scan(f)
        .group_by(&["k"], &[("v", AggKind::Sum)])
        .filter(col("k").ge(lit_i64(0)));
    let text = plan.clone().optimize().to_string();
    assert!(text.contains("filter=(col(k) >= 0)"), "{text}");
}

#[test]
fn filter_moves_below_withcolumns_unless_it_uses_new_columns() {
    let df = frame(
        ("a", "b"),
        &vec![(Some(1), Some(10)), (Some(2), Some(20)), (Some(3), None)],
    );
    let plan = LogicalPlan::scan(df)
        .with_columns(vec![(col("a") + col("b")).alias("s"), col("b").alias("a2")])
        .filter(col("a").gt(lit_i64(1)).and(col("s").gt(lit_i64(0))));
    let text = plan.clone().optimize().to_string();
    assert!(
        at(&text, "FILTER (col(s) > 0)") < at(&text, "WITH_COLUMNS"),
        "{text}"
    );
    assert!(text.contains("filter=(col(a) > 1)"), "{text}");
    assert_eq!(
        logical(&plan.collect().unwrap()),
        logical(&plan.optimize().collect().unwrap())
    );
    // a replaced column counts as new: the predicate sees the new values
    let df = frame(("a", "b"), &vec![(Some(1), Some(10)), (Some(2), Some(20))]);
    let plan = LogicalPlan::scan(df)
        .with_columns(vec![(col("a") + lit_i64(5)).alias("a")])
        .filter(col("a").gt(lit_i64(6)));
    let text = plan.clone().optimize().to_string();
    assert!(
        at(&text, "FILTER (col(a) > 6)") < at(&text, "WITH_COLUMNS"),
        "{text}"
    );
    assert_eq!(plan.collect().unwrap().height(), 1);
    assert_eq!(plan.optimize().collect().unwrap().height(), 1);
    // NOT over a non-boolean can fail on data: nothing moves below it
    let df = frame(("a", "b"), &vec![(Some(1), Some(10))]);
    let plan = LogicalPlan::scan(df)
        .with_columns(vec![col("b").not().alias("n")])
        .filter(col("a").gt(lit_i64(5)));
    let text = plan.clone().optimize().to_string();
    assert!(at(&text, "FILTER") < at(&text, "WITH_COLUMNS"), "{text}");
    assert!(plan.clone().collect().is_err());
    assert!(plan.optimize().collect().is_err());
}

#[test]
fn filter_never_moves_below_limit() {
    // The first two rows fail the predicate: pushing the filter below the
    // limit would return the (3, 30) row instead of nothing.
    let df = frame(
        ("a", "b"),
        &vec![(Some(0), Some(1)), (Some(0), Some(2)), (Some(3), Some(30))],
    );
    let plan = LogicalPlan::scan(df)
        .limit(2)
        .filter(col("a").gt(lit_i64(1)));
    let opt = plan.clone().optimize();
    let text = opt.to_string();
    assert!(at(&text, "FILTER") < at(&text, "LIMIT"), "{text}");
    assert!(!text.contains("filter=("), "{text}");
    assert_eq!(plan.collect().unwrap().height(), 0);
    assert_eq!(opt.collect().unwrap().height(), 0);
}

#[test]
fn rules_fire_on_join_and_scan() {
    let l = frame(("a", "b"), &vec![(Some(1), Some(2)), (Some(2), None)]);
    let r = frame(("a", "c"), &vec![(Some(1), Some(7))]);
    let plan = LogicalPlan::scan(l)
        .join(LogicalPlan::scan(r), &["a"], JoinType::Left)
        .filter(col("b").gt(lit_i64(0)).and(col("c").gt(lit_i64(0))))
        .select(&["a", "c"])
        .sort(&["a"], &[false]);
    let text = plan.clone().optimize().to_string();
    // The left-only conjunct moves into the left scan; the right-only one must
    // stay above a LEFT join.
    assert!(text.contains("filter=(col(b) > 0)"), "{text}");
    assert!(text.contains("FILTER (col(c) > 0)"), "{text}");
    assert_eq!(
        logical(&plan.collect().unwrap()),
        logical(&plan.clone().optimize().collect().unwrap())
    );
}

#[test]
fn not_and_division_fold_only_when_defined() {
    use tessera_lazy::lit_bool;
    let df = frame(("a", "b"), &vec![(Some(1), Some(2))]);
    let plan = LogicalPlan::scan(df)
        .filter(lit_bool(true).not().or(col("a").gt(lit_i64(0))))
        .filter((lit_i64(6) / lit_i64(3)).equals(lit_i64(2)))
        .filter((lit_i64(6) / lit_i64(0)).equals(lit_i64(0)));
    let text = plan.clone().optimize().to_string();
    // NOT true folds to false, 6 / 3 folds to 2 (so the filter is removed),
    // but 6 / 0 is null and must stay an expression.
    assert!(!text.contains("NOT"), "{text}");
    assert!(text.contains("/ 0"), "{text}");
    assert_eq!(
        logical(&plan.collect().unwrap()),
        logical(&plan.optimize().collect().unwrap())
    );
}

#[test]
fn csv_projection_and_filter_pushdown() {
    let dir = std::env::temp_dir().join(format!("tessera_lazy_csv_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.csv");
    std::fs::write(
        &path,
        "a,b,c,d\n1,x,10,0.5\n2,y,20,1.5\n3,x,30,2.5\n,z,40,3.5\n",
    )
    .unwrap();
    let plan = LogicalPlan::scan_csv(&path, CsvOptions::new())
        .unwrap()
        .filter(col("c").gt(lit_i64(10)))
        .group_by(&["b"], &[("c", AggKind::Sum)])
        .sort(&["b"], &[false]);
    assert_eq!(plan.columns().unwrap(), ["b", "c_sum"]);
    let opt = plan.clone().optimize();
    let text = opt.to_string();
    assert!(
        text.contains("SCAN CSV") && text.contains("[b, c]"),
        "{text}"
    );
    let (x, y) = (plan.collect().unwrap(), opt.collect().unwrap());
    assert_eq!(logical(&x), logical(&y));
    assert_eq!(x.shape(), (3, 2));
    let schema = plan.schema().unwrap();
    assert_eq!(schema[1].0, "c_sum");
    std::fs::remove_dir_all(&dir).ok();
}

/// A conjunct that can fail on data (`NOT` over a non-boolean) must stay where
/// it is: moving it below a group-by makes it run before the group-by's own
/// error, so the surfaced error changes even though both plans fail. (The
/// matching `!p.may_fail()` guard in the with_columns rule is equivalent: see the
/// comment there.)
#[test]
fn failing_predicate_keeps_its_place_relative_to_node_errors() {
    let df = frame(("a", "b"), &vec![(Some(1), Some(10)), (Some(2), Some(20))]);
    let message = |plan: LogicalPlan| plan.collect().unwrap_err().to_string();

    // group-by whose aggregate column is missing fails first
    let plan = LogicalPlan::scan(df.clone())
        .group_by(&["a"], &[("missing", AggKind::Sum)])
        .filter(col("a").not());
    let (before, after) = (message(plan.clone()), message(plan.optimize()));
    assert_eq!(before, after);
    assert!(before.contains("missing"), "{before}");
}

/// Regression (fuzz `expr_eval`): `p1 AND p2` where `p2` can fail on data
/// (`NOT` over a non-boolean) and `p1` is movable. `AND` evaluates both sides
/// on every row, so the whole predicate fails when `p2` meets a non-boolean.
/// Splitting it and moving only `p1` below the node first removed the rows
/// that `p2` would have failed on, and the optimised plan succeeded. A
/// predicate with a failing conjunct must not be split at all.
#[test]
fn failing_conjunct_is_not_split_off_from_a_movable_one() {
    let df = frame(("a", "b"), &vec![(Some(1), Some(10)), (Some(2), Some(20))]);
    let right = frame(("a", "c"), &vec![(Some(1), Some(5)), (Some(2), Some(6))]);
    // p1 is false on every row; p2 fails on the first non-null value.
    let p1 = col("a").gt(lit_i64(100));
    let p2 = col("a").not().or(col("a").gt(lit_i64(0)));
    let predicate = p1.and(p2);

    let plans = [
        (
            "join",
            LogicalPlan::scan(df.clone())
                .join(LogicalPlan::scan(right), &["a"], JoinType::Inner)
                .filter(predicate.clone()),
        ),
        (
            "group_by",
            LogicalPlan::scan(df.clone())
                .group_by(&["a"], &[("b", AggKind::Sum)])
                .filter(predicate.clone()),
        ),
        (
            "with_columns",
            LogicalPlan::scan(df)
                .with_columns(vec![(col("b") + lit_i64(1)).alias("n")])
                .filter(predicate),
        ),
    ];
    for (name, plan) in plans {
        assert!(plan.collect().is_err(), "{name}: unoptimized must fail");
        assert!(
            plan.clone().optimize().collect().is_err(),
            "{name}: optimized must fail"
        );
    }
}

/// A failing filter that sits above another filter only sees the rows the
/// first one kept; merging them with `AND` would show it every row.
#[test]
fn failing_filter_is_not_merged_into_an_earlier_filter() {
    let df = frame(("a", "b"), &vec![(Some(1), Some(10)), (Some(2), Some(20))]);
    let never = col("a").gt(lit_i64(100));
    let fails = col("a").not().or(col("a").gt(lit_i64(0)));
    let filters = LogicalPlan::scan(df.clone())
        .filter(never.clone())
        .filter(fails.clone());
    let scan_pred = LogicalPlan::scan(df).filter(never).filter(fails);
    for (name, plan) in [("filters", filters), ("scan", scan_pred)] {
        let plain = plan.collect().map(|d| d.shape());
        let optimized = plan.optimize().collect().map(|d| d.shape());
        assert_eq!(plain.is_ok(), optimized.is_ok(), "{name}");
        assert_eq!(plain.unwrap(), (0, 2), "{name}");
    }
}
