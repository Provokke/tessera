//! Rule-based optimiser.
//!
//! Rules (applied bottom-up to a fixpoint, then one top-down pruning pass):
//!
//! * constant folding and integer algebraic simplification (`x + 0`, `x * 1`,
//!   only when `x` is known to be `Int64`) of every expression;
//! * `Filter(true)` removal;
//! * filter merging (`Filter(Filter(x, p), q)` => `Filter(x, p AND q)`);
//! * predicate pushdown below `Select` (when the predicate only uses selected
//!   columns), below `Sort`, into an in-memory `Scan`, and into the sides of a
//!   `Join` (conjuncts that use only left columns go left for inner/left joins;
//!   conjuncts that use only non-clashing right columns go right for
//!   inner/right joins; a full outer join pushes nothing);
//! * predicate pushdown below `GroupBy` (conjuncts over key columns only) and
//!   below `WithColumns` (conjuncts that use no created column); a `Limit` is
//!   never crossed;
//! * projection pushdown: the columns a plan needs are pushed into in-memory
//!   scans and into the CSV parser.
//!
//! Predicates are only merged, split or moved when they are statically
//! boolean ([`Expr::is_boolean`]); otherwise `AND` would turn the error of a
//! non-boolean filter into an all-false mask. A predicate that can fail on
//! data (`Expr::may_fail`) is never merged into an earlier filter (it would
//! see rows the earlier filter removed) and never split apart from its
//! siblings.
use std::collections::BTreeSet;

use tessera_core::{CoreError, DataFrame, DataType, JoinType};

use crate::expr::{Expr, Op};
use crate::plan::{LogicalPlan, Schema};

impl LogicalPlan {
    /// Apply the optimiser rules to a fixpoint, then prune unused columns.
    pub fn optimize(self) -> Self {
        let mut plan = self;
        loop {
            let (next, changed) = rewrite(plan);
            plan = next;
            if !changed {
                break;
            }
        }
        prune(plan, None)
    }
}

fn boxed(p: LogicalPlan) -> Box<LogicalPlan> {
    Box::new(p)
}

/// Rewrite the children of a node with `f`; returns whether any changed.
fn map_children(
    plan: LogicalPlan,
    f: &mut dyn FnMut(LogicalPlan) -> (LogicalPlan, bool),
) -> (LogicalPlan, bool) {
    match plan {
        LogicalPlan::Scan { .. } | LogicalPlan::ScanCsv { .. } => (plan, false),
        LogicalPlan::Filter { input, predicate } => {
            let (i, c) = f(*input);
            (
                LogicalPlan::Filter {
                    input: boxed(i),
                    predicate,
                },
                c,
            )
        }
        LogicalPlan::Select { input, columns } => {
            let (i, c) = f(*input);
            (
                LogicalPlan::Select {
                    input: boxed(i),
                    columns,
                },
                c,
            )
        }
        LogicalPlan::WithColumns { input, exprs } => {
            let (i, c) = f(*input);
            (
                LogicalPlan::WithColumns {
                    input: boxed(i),
                    exprs,
                },
                c,
            )
        }
        LogicalPlan::GroupBy { input, keys, aggs } => {
            let (i, c) = f(*input);
            (
                LogicalPlan::GroupBy {
                    input: boxed(i),
                    keys,
                    aggs,
                },
                c,
            )
        }
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => {
            let (i, c) = f(*input);
            (
                LogicalPlan::Sort {
                    input: boxed(i),
                    by,
                    descending,
                },
                c,
            )
        }
        LogicalPlan::Limit { input, n } => {
            let (i, c) = f(*input);
            (LogicalPlan::Limit { input: boxed(i), n }, c)
        }
        LogicalPlan::Join {
            left,
            right,
            left_on,
            right_on,
            how,
        } => {
            let (l, c1) = f(*left);
            let (r, c2) = f(*right);
            (
                LogicalPlan::Join {
                    left: boxed(l),
                    right: boxed(r),
                    left_on,
                    right_on,
                    how,
                },
                c1 || c2,
            )
        }
    }
}

/// One bottom-up rewrite pass. Returns whether anything changed.
fn rewrite(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let (plan, c) = map_children(plan, &mut rewrite);
    let (plan, c2) = rewrite_node(plan);
    (plan, c || c2)
}

fn df_schema(df: &DataFrame) -> Schema {
    df.columns()
        .iter()
        .map(|c| (c.name().to_owned(), c.data_type()))
        .collect()
}

fn lookup(schema: &Schema, c: &str) -> Result<DataType, CoreError> {
    schema
        .iter()
        .find(|(n, _)| n == c)
        .map(|(_, d)| *d)
        .ok_or_else(|| CoreError::ColumnNotFound(c.to_owned()))
}

/// Constant folding and algebraic simplification. `schema` enables the
/// type-dependent identities.
fn simplify(e: Expr, schema: Option<&Schema>) -> (Expr, bool) {
    match e {
        Expr::Alias(inner, n) => {
            let (i, c) = simplify(*inner, schema);
            (Expr::Alias(Box::new(i), n), c)
        }
        Expr::Binary(l, op, r) => {
            let (l, cl) = simplify(*l, schema);
            let (r, cr) = simplify(*r, schema);
            let node = Expr::Binary(Box::new(l), op, Box::new(r));
            if let Some(s) = node.fold_constant() {
                return (Expr::Lit(s), true);
            }
            let Expr::Binary(l, op, r) = node else {
                return (node, cl || cr);
            };
            let is_int64 = |x: &Expr| {
                schema.is_some_and(|sc| x.dtype_in(&|c| lookup(sc, c)) == Ok(DataType::Int64))
            };
            let lit = |x: &Expr, v: i64| matches!(x, Expr::Lit(tessera_core::Scalar::Int64(k)) if *k == v);
            let keep_left = match op {
                Op::Add => lit(&r, 0) && is_int64(&l),
                Op::Sub => lit(&r, 0) && is_int64(&l),
                Op::Mul => lit(&r, 1) && is_int64(&l),
                _ => false,
            };
            if keep_left {
                return (*l, true);
            }
            let keep_right = match op {
                Op::Add => lit(&l, 0) && is_int64(&r),
                Op::Mul => lit(&l, 1) && is_int64(&r),
                _ => false,
            };
            if keep_right {
                return (*r, true);
            }
            (Expr::Binary(l, op, r), cl || cr)
        }
        Expr::Not(inner) => {
            let (i, c) = simplify(*inner, schema);
            let node = Expr::Not(Box::new(i));
            match node.fold_constant() {
                Some(s) => (Expr::Lit(s), true),
                None => (node, c),
            }
        }
        other => (other, false),
    }
}

/// Split `a AND b AND c` into conjuncts, but only through `AND`s whose both
/// sides are statically boolean.
fn conjuncts(e: Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::Binary(l, Op::And, r) if l.is_boolean() && r.is_boolean() => {
            conjuncts(*l, out);
            conjuncts(*r, out);
        }
        other => out.push(other),
    }
}

fn and_all(parts: Vec<Expr>) -> Option<Expr> {
    parts.into_iter().reduce(Expr::and)
}

fn subset(cols: &BTreeSet<String>, names: &[String]) -> bool {
    cols.iter().all(|c| names.contains(c))
}

fn rewrite_node(plan: LogicalPlan) -> (LogicalPlan, bool) {
    match plan {
        LogicalPlan::Filter { input, predicate } => rewrite_filter(*input, predicate),
        LogicalPlan::WithColumns { input, exprs } => {
            let schema = input.known_schema();
            let mut changed = false;
            let exprs = exprs
                .into_iter()
                .map(|(n, e)| {
                    let (e, c) = simplify(e, schema.as_ref());
                    changed |= c;
                    (n, e)
                })
                .collect();
            (LogicalPlan::WithColumns { input, exprs }, changed)
        }
        LogicalPlan::Scan {
            df,
            projection,
            predicate: Some(p),
        } => {
            let (p, c) = simplify(p, Some(&df_schema(&df)));
            (
                LogicalPlan::Scan {
                    df,
                    projection,
                    predicate: Some(p),
                },
                c,
            )
        }
        other => (other, false),
    }
}

fn rewrite_filter(input: LogicalPlan, predicate: Expr) -> (LogicalPlan, bool) {
    let schema = input.known_schema();
    let (predicate, simplified) = simplify(predicate, schema.as_ref());
    let rebuilt = |input: LogicalPlan, predicate: Expr, changed: bool| {
        (
            LogicalPlan::Filter {
                input: boxed(input),
                predicate,
            },
            changed,
        )
    };
    if predicate == Expr::Lit(tessera_core::Scalar::Boolean(true)) {
        return (input, true);
    }
    match input {
        // Filter merging.
        LogicalPlan::Filter {
            input: inner,
            predicate: p,
        } if p.is_boolean() && predicate.is_boolean() && !predicate.may_fail() => (
            LogicalPlan::Filter {
                input: inner,
                predicate: p.and(predicate),
            },
            true,
        ),
        // Pushdown below a projection (the predicate must only use selected columns).
        LogicalPlan::Select {
            input: inner,
            columns,
        } if subset(&predicate.columns(), &columns) => (
            LogicalPlan::Select {
                input: boxed(LogicalPlan::Filter {
                    input: inner,
                    predicate,
                }),
                columns,
            },
            true,
        ),
        // Pushdown below a (stable) sort.
        LogicalPlan::Sort {
            input: inner,
            by,
            descending,
        } => (
            LogicalPlan::Sort {
                input: boxed(LogicalPlan::Filter {
                    input: inner,
                    predicate,
                }),
                by,
                descending,
            },
            true,
        ),
        // Pushdown into an in-memory scan.
        LogicalPlan::Scan {
            df,
            projection,
            predicate: existing,
        } if predicate.is_boolean()
            && !predicate.may_fail()
            && existing.as_ref().is_none_or(Expr::is_boolean)
            && projection
                .as_ref()
                .is_none_or(|p| subset(&predicate.columns(), p)) =>
        {
            let merged = match existing {
                Some(e) => e.and(predicate),
                None => predicate,
            };
            (
                LogicalPlan::Scan {
                    df,
                    projection,
                    predicate: Some(merged),
                },
                true,
            )
        }
        // Pushdown into join sides. A side's rows survive the join unchanged
        // only when the join keeps all of them: left-only conjuncts go left
        // for inner/left joins, right-only conjuncts go right for inner/right
        // joins; a full outer join pushes nothing.
        LogicalPlan::Join {
            left,
            right,
            left_on,
            right_on,
            how,
        } => {
            let (names_l, names_r) = match (left.columns(), right.columns()) {
                (Ok(l), Ok(r)) => (l, r),
                _ => {
                    return rebuilt(
                        LogicalPlan::Join {
                            left,
                            right,
                            left_on,
                            right_on,
                            how,
                        },
                        predicate,
                        simplified,
                    )
                }
            };
            // Right key columns that the join drops (same name as the left key).
            let dropped: Vec<&String> = left_on
                .iter()
                .zip(&right_on)
                .filter(|(a, b)| a == b)
                .map(|(_, b)| b)
                .collect();
            let right_only: Vec<String> = names_r
                .iter()
                .filter(|n| !dropped.contains(n) && !names_l.contains(n))
                .cloned()
                .collect();
            let can_left = matches!(how, JoinType::Inner | JoinType::Left);
            let can_right = matches!(how, JoinType::Inner | JoinType::Right);
            let mut parts = Vec::new();
            conjuncts(predicate, &mut parts);
            let split_ok = splittable(&parts);
            let (mut to_left, mut to_right, mut keep) = (Vec::new(), Vec::new(), Vec::new());
            for p in parts {
                let cols = p.columns();
                let movable = split_ok && p.is_boolean() && !cols.is_empty();
                if movable && can_left && subset(&cols, &names_l) {
                    to_left.push(p);
                } else if movable && can_right && subset(&cols, &right_only) {
                    to_right.push(p);
                } else {
                    keep.push(p);
                }
            }
            let pushed = !to_left.is_empty() || !to_right.is_empty();
            let wrap = |side: Box<LogicalPlan>, parts: Vec<Expr>| match and_all(parts) {
                Some(p) => boxed(LogicalPlan::Filter {
                    input: side,
                    predicate: p,
                }),
                None => side,
            };
            let join = LogicalPlan::Join {
                left: wrap(left, to_left),
                right: wrap(right, to_right),
                left_on,
                right_on,
                how,
            };
            match and_all(keep) {
                Some(p) => rebuilt(join, p, pushed || simplified),
                None => (join, true),
            }
        }
        // Pushdown below a group-by: conjuncts that use only key columns remove
        // whole groups, so they can filter the input instead. Group order
        // (first occurrence) is unaffected. `-0.0` and `0.0` share a group, so a
        // predicate with a division (`1 / k` tells them apart) is only moved
        // when every key it uses is known not to be a float.
        LogicalPlan::GroupBy {
            input: inner,
            keys,
            aggs,
        } => {
            let schema = inner.known_schema();
            let mut parts = Vec::new();
            conjuncts(predicate, &mut parts);
            let split_ok = splittable(&parts);
            let (mut down, mut keep) = (Vec::new(), Vec::new());
            for p in parts {
                let cols = p.columns();
                if split_ok
                    && p.is_boolean()
                    && !cols.is_empty()
                    && subset(&cols, &keys)
                    && (!p.has_division() || no_float(&cols, schema.as_ref()))
                {
                    down.push(p);
                } else {
                    keep.push(p);
                }
            }
            let pushed = !down.is_empty();
            let inner = match and_all(down) {
                Some(p) => boxed(LogicalPlan::Filter {
                    input: inner,
                    predicate: p,
                }),
                None => inner,
            };
            let grouped = LogicalPlan::GroupBy {
                input: inner,
                keys,
                aggs,
            };
            match and_all(keep) {
                Some(p) => rebuilt(grouped, p, pushed || simplified),
                None => (grouped, true),
            }
        }
        // Pushdown below with_columns: conjuncts that do not use a new or
        // replaced column see the same values below. The expressions must not
        // be able to fail on data (their error could vanish when rows are
        // removed) and must only use columns that exist. The predicate itself
        // is only split when no conjunct can fail (see `splittable`).
        LogicalPlan::WithColumns {
            input: inner,
            exprs,
        } if inner
            .columns()
            .is_ok_and(|names| computable(&names, &exprs)) =>
        {
            let created: BTreeSet<&str> = exprs.iter().map(|(n, _)| n.as_str()).collect();
            let mut parts = Vec::new();
            conjuncts(predicate, &mut parts);
            let split_ok = splittable(&parts);
            let (mut down, mut keep) = (Vec::new(), Vec::new());
            for p in parts {
                let cols = p.columns();
                if split_ok
                    && p.is_boolean()
                    && !cols.is_empty()
                    && cols.iter().all(|c| !created.contains(c.as_str()))
                {
                    down.push(p);
                } else {
                    keep.push(p);
                }
            }
            let pushed = !down.is_empty();
            let inner = match and_all(down) {
                Some(p) => boxed(LogicalPlan::Filter {
                    input: inner,
                    predicate: p,
                }),
                None => inner,
            };
            let with = LogicalPlan::WithColumns {
                input: inner,
                exprs,
            };
            match and_all(keep) {
                Some(p) => rebuilt(with, p, pushed || simplified),
                None => (with, true),
            }
        }
        // A filter must NOT move below a limit: it would change which rows
        // the limit keeps.
        other @ LogicalPlan::Limit { .. } => rebuilt(other, predicate, simplified),
        other => rebuilt(other, predicate, simplified),
    }
}

/// Whether a predicate's conjuncts may be moved apart. `AND` evaluates both
/// sides on every row, so a conjunct that can fail on data (`NOT` over a
/// non-boolean) fails the whole predicate. If its sibling were moved below a
/// node and removed rows first, the failing conjunct would see fewer rows (or
/// none) and the error would vanish, so nothing is split off such a predicate.
fn splittable(parts: &[Expr]) -> bool {
    !parts.iter().any(Expr::may_fail)
}

/// Every key column's type is known and is not `Float64`.
fn no_float(cols: &BTreeSet<String>, schema: Option<&Schema>) -> bool {
    schema.is_some_and(|sc| {
        cols.iter()
            .all(|c| lookup(sc, c).is_ok_and(|d| d != DataType::Float64))
    })
}

/// Whether every expression only uses columns that exist when it is evaluated
/// (input columns plus the ones created before it).
fn computable(input_names: &[String], exprs: &[(String, Expr)]) -> bool {
    let mut avail: Vec<&str> = input_names.iter().map(String::as_str).collect();
    for (n, e) in exprs {
        if e.may_fail() || !e.columns().iter().all(|c| avail.contains(&c.as_str())) {
            return false;
        }
        avail.push(n);
    }
    true
}

fn names(v: &[String]) -> BTreeSet<String> {
    v.iter().cloned().collect()
}

/// Top-down projection pushdown. `req` is the set of columns the parent
/// needs (`None` = all). Column order of intermediate results may change, but
/// every pruning parent addresses columns by name and the root is never pruned.
fn prune(plan: LogicalPlan, req: Option<BTreeSet<String>>) -> LogicalPlan {
    let with = |extra: BTreeSet<String>| req.clone().map(|r| r.union(&extra).cloned().collect());
    match plan {
        LogicalPlan::Scan {
            df,
            projection,
            predicate,
        } => {
            let current: Vec<String> = projection
                .clone()
                .unwrap_or_else(|| df.column_names().iter().map(|s| (*s).to_owned()).collect());
            let projection = match &req {
                Some(r) => {
                    let keep: Vec<String> =
                        current.iter().filter(|c| r.contains(*c)).cloned().collect();
                    if keep.is_empty() || keep.len() == current.len() {
                        projection
                    } else {
                        Some(keep)
                    }
                }
                None => projection,
            };
            LogicalPlan::Scan {
                df,
                projection,
                predicate,
            }
        }
        LogicalPlan::ScanCsv {
            path,
            mut options,
            header,
        } => {
            if let Some(r) = &req {
                let current = options.projection.clone().unwrap_or_else(|| header.clone());
                let keep: Vec<String> =
                    current.iter().filter(|c| r.contains(*c)).cloned().collect();
                if !keep.is_empty() && keep.len() != current.len() {
                    options.projection = Some(keep);
                }
            }
            LogicalPlan::ScanCsv {
                path,
                options,
                header,
            }
        }
        LogicalPlan::Filter { input, predicate } => {
            let r = with(predicate.columns());
            LogicalPlan::Filter {
                input: boxed(prune(*input, r)),
                predicate,
            }
        }
        LogicalPlan::Select { input, columns } => {
            let r = Some(names(&columns));
            LogicalPlan::Select {
                input: boxed(prune(*input, r)),
                columns,
            }
        }
        LogicalPlan::WithColumns { input, exprs } => {
            let used: BTreeSet<String> = exprs.iter().flat_map(|(_, e)| e.columns()).collect();
            let r = with(used);
            LogicalPlan::WithColumns {
                input: boxed(prune(*input, r)),
                exprs,
            }
        }
        LogicalPlan::GroupBy { input, keys, aggs } => {
            let mut r = names(&keys);
            r.extend(aggs.iter().map(|(c, _)| c.clone()));
            LogicalPlan::GroupBy {
                input: boxed(prune(*input, Some(r))),
                keys,
                aggs,
            }
        }
        LogicalPlan::Sort {
            input,
            by,
            descending,
        } => {
            let r = with(names(&by));
            LogicalPlan::Sort {
                input: boxed(prune(*input, r)),
                by,
                descending,
            }
        }
        LogicalPlan::Limit { input, n } => LogicalPlan::Limit {
            input: boxed(prune(*input, req)),
            n,
        },
        LogicalPlan::Join {
            left,
            right,
            left_on,
            right_on,
            how,
        } => {
            // Output names equal input names unless a right column clashes with
            // a left one (then it is renamed), so prune only without clashes.
            let dropped: Vec<&String> = left_on
                .iter()
                .zip(&right_on)
                .filter(|(a, b)| a == b)
                .map(|(_, b)| b)
                .collect();
            let sides = match (&req, left.columns(), right.columns()) {
                (Some(r), Ok(l), Ok(rc))
                    if !rc.iter().any(|n| !dropped.contains(&n) && l.contains(n)) =>
                {
                    let mut need = r.clone();
                    need.extend(left_on.iter().cloned());
                    need.extend(right_on.iter().cloned());
                    Some(need)
                }
                _ => None,
            };
            LogicalPlan::Join {
                left: boxed(prune(*left, sides.clone())),
                right: boxed(prune(*right, sides)),
                left_on,
                right_on,
                how,
            }
        }
    }
}
