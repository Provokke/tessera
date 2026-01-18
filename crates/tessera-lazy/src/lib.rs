//! # tessera-lazy
//!
//! A small lazy query layer:
//!
//! * [`Expr`]: columns, literals, `+ - *`, `== < >`, `AND`/`OR`, aliases.
//!   Evaluation is vectorized over the `tessera-compute` kernels (comparisons,
//!   arithmetic, boolean ops, column-vs-column and column-vs-literal) and falls
//!   back to a row-wise reference evaluator for combinations the kernels do
//!   not cover.
//! * [`LogicalPlan`]: `Scan` (in-memory), `ScanCsv`, `Filter`, `Select`,
//!   `WithColumns`, `GroupBy`, `Join` (inner/left), `Sort`, `Limit`, with schema
//!   inference ([`LogicalPlan::schema`], [`LogicalPlan::columns`]).
//! * An optimiser (see [`optimizer`] docs): constant folding, filter merging,
//!   predicate pushdown (below `Select`/`Sort`, into scans and join sides) and
//!   projection pushdown (into scans and the CSV parser).
//! * Execution: `GroupBy` and `Join` run the parallel partitioned group-by and
//!   hash join of `tessera-core`.
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

mod expr;
pub mod optimizer;
mod plan;

pub use expr::{col, lit_bool, lit_f64, lit_i64, lit_str, Expr, Op};
pub use plan::{LogicalPlan, Schema};

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_arrow::{Array, PrimitiveArray};
    use tessera_core::{AggKind, DataFrame, JoinType, Scalar, Series};

    /// Logical contents (values under null slots are unspecified).
    fn rows(df: &DataFrame) -> Vec<(String, Vec<Scalar>)> {
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

    fn df() -> DataFrame {
        DataFrame::new(vec![
            Series::new(
                "a",
                Array::Int64(PrimitiveArray::from_options([
                    Some(1),
                    Some(5),
                    None,
                    Some(9),
                ])),
            ),
            Series::new(
                "b",
                Array::Int64(PrimitiveArray::from_options([
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn optimized_plan_shape() {
        let plan = LogicalPlan::scan(df())
            .filter(col("a").gt(lit_i64(1)))
            .select(&["a"])
            .filter(col("a").lt(lit_i64(9)));
        let opt = plan.clone().optimize();
        assert_eq!(
            opt.to_string(),
            "SELECT [a]\n  SCAN [a, b] projection=[a] filter=((col(a) > 1) AND (col(a) < 9))\n"
        );
        assert_eq!(
            rows(&plan.collect().unwrap()),
            rows(&opt.collect().unwrap())
        );
        assert_eq!(opt.collect().unwrap().height(), 1);
        assert!(plan.explain().contains("== optimized =="));
    }

    #[test]
    fn eval_arith_and_nulls() {
        let r = ((col("a") + col("b")) * lit_i64(2))
            .evaluate(&df())
            .unwrap();
        assert_eq!(r.scalar(0), Scalar::Int64(6));
        assert!(r.scalar(2).is_null());
        assert!(LogicalPlan::scan(df()).filter(col("a")).collect().is_err());
    }

    #[test]
    fn constant_folding_and_identities() {
        let plan = LogicalPlan::scan(df())
            .with_columns(vec![((col("a") + lit_i64(0)) * lit_i64(1)
                + (lit_i64(2) * lit_i64(3)))
            .alias("c")])
            .filter(lit_i64(1).lt(lit_i64(2)));
        let opt = plan.clone().optimize();
        assert_eq!(
            opt.to_string(),
            "WITH_COLUMNS [(col(a) + 6) AS c]
  SCAN [a, b]
"
        );
        assert_eq!(
            rows(&plan.collect().unwrap()),
            rows(&opt.collect().unwrap())
        );
    }

    #[test]
    fn projection_and_join_pushdown() {
        let right = DataFrame::new(vec![
            Series::new(
                "a",
                Array::Int64(PrimitiveArray::from_options([Some(1), Some(9)])),
            ),
            Series::new(
                "z",
                Array::Int64(PrimitiveArray::from_options([Some(10), Some(90)])),
            ),
            Series::new(
                "w",
                Array::Int64(PrimitiveArray::from_options([Some(0), Some(0)])),
            ),
        ])
        .unwrap();
        let plan = LogicalPlan::scan(df())
            .join(LogicalPlan::scan(right), &["a"], JoinType::Inner)
            .filter(col("b").gt(lit_i64(1)).and(col("z").gt(lit_i64(10))))
            .select(&["a", "z"]);
        let opt = plan.clone().optimize();
        let text = opt.to_string();
        assert!(
            text.contains("projection=[a] filter=(col(b) > 1)"),
            "{text}"
        );
        assert!(
            text.contains("projection=[a, z] filter=(col(z) > 10)"),
            "{text}"
        );
        assert_eq!(
            rows(&plan.collect().unwrap()),
            rows(&opt.collect().unwrap())
        );
        assert_eq!(opt.collect().unwrap().shape(), (0, 2));
    }

    #[test]
    fn group_by_sort_limit_schema() {
        let plan = LogicalPlan::scan(df())
            .group_by(&["b"], &[("a", AggKind::Sum), ("a", AggKind::Mean)])
            .sort(&["b"], &[false])
            .limit(2);
        let schema = plan.schema().unwrap();
        let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "a_sum", "a_mean"]);
        assert_eq!(schema[2].1, tessera_core::DataType::Float64);
        let out = plan.optimize().collect().unwrap();
        assert_eq!(out.shape(), (2, 3));
        assert_eq!(out.column("b").unwrap().get(0), Scalar::Int64(2));
    }
}
