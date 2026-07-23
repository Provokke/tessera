//! Whole-query property test: random pipelines of Filter / WithColumns /
//! Select / Join / Sort / GroupBy / Limit are run unoptimised and optimised
//! and compared against a naive row-based reference over `Vec<Option<i64>>`
//! tables (no tessera kernels involved in the reference).
use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray};
use tessera_core::{AggKind, DataFrame, JoinType, Scalar, Series};
use tessera_lazy::{col, lit_i64, Expr, LogicalPlan};

type Cell = Option<i64>;

/// Row-major reference table.
#[derive(Clone, Debug, PartialEq)]
struct Tab {
    names: Vec<String>,
    rows: Vec<Vec<Cell>>,
}

impl Tab {
    fn idx(&self, n: &str) -> usize {
        self.names.iter().position(|x| x == n).unwrap()
    }
    fn has(&self, n: &str) -> bool {
        self.names.iter().any(|x| x == n)
    }
    fn to_frame(&self) -> DataFrame {
        DataFrame::new(
            self.names
                .iter()
                .enumerate()
                .map(|(c, n)| {
                    Series::new(
                        n.clone(),
                        Array::Int64(PrimitiveArray::from_options(self.rows.iter().map(|r| r[c]))),
                    )
                })
                .collect(),
        )
        .unwrap()
    }
    fn from_frame(df: &DataFrame) -> Tab {
        let names = df.column_names().iter().map(|s| s.to_string()).collect();
        let rows = (0..df.height())
            .map(|i| {
                df.columns()
                    .iter()
                    .map(|c| match c.get(i) {
                        Scalar::Int64(v) => Some(v),
                        Scalar::Null(_) => None,
                        other => panic!("unexpected scalar {other:?}"),
                    })
                    .collect()
            })
            .collect();
        Tab { names, rows }
    }
}

/// Tiny deterministic generator so the whole query derives from one seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Debug)]
enum E {
    Col(String),
    Lit(i64),
    /// 0 add, 1 sub, 2 mul, 3 gt, 4 lt, 5 eq, 6 and, 7 or, 8 div (null by
    /// zero), 9 ge, 10 le, 11 ne
    Bin(Box<E>, u8, Box<E>),
    /// Logical negation of a boolean (0/1) operand.
    Not(Box<E>),
}

fn gen_int(r: &mut Rng, names: &[String], depth: u8) -> E {
    if depth == 0 || r.below(3) == 0 {
        return if r.below(4) == 0 {
            E::Lit(r.below(7) as i64 - 3)
        } else {
            E::Col(names[r.below(names.len())].clone())
        };
    }
    E::Bin(
        Box::new(gen_int(r, names, depth - 1)),
        [0, 1, 2, 8][r.below(4)],
        Box::new(gen_int(r, names, depth - 1)),
    )
}

fn gen_bool(r: &mut Rng, names: &[String], depth: u8) -> E {
    if depth == 0 || r.below(2) == 0 {
        return E::Bin(
            Box::new(gen_int(r, names, 1)),
            [3, 4, 5, 9, 10, 11][r.below(6)],
            Box::new(gen_int(r, names, 1)),
        );
    }
    if r.below(4) == 0 {
        return E::Not(Box::new(gen_bool(r, names, depth - 1)));
    }
    E::Bin(
        Box::new(gen_bool(r, names, depth - 1)),
        6 + r.below(2) as u8,
        Box::new(gen_bool(r, names, depth - 1)),
    )
}

fn to_expr(e: &E) -> Expr {
    match e {
        E::Col(c) => col(c),
        E::Lit(k) => lit_i64(*k),
        E::Not(e) => to_expr(e).not(),
        E::Bin(l, op, r) => {
            let (l, r) = (to_expr(l), to_expr(r));
            match op {
                0 => l + r,
                1 => l - r,
                2 => l * r,
                3 => l.gt(r),
                4 => l.lt(r),
                5 => l.equals(r),
                6 => l.and(r),
                7 => l.or(r),
                8 => l / r,
                9 => l.ge(r),
                10 => l.le(r),
                _ => l.not_equals(r),
            }
        }
    }
}

/// Null-propagating reference semantics; booleans are 0/1.
fn eval(e: &E, t: &Tab, row: &[Cell]) -> Cell {
    match e {
        E::Col(c) => row[t.idx(c)],
        E::Lit(k) => Some(*k),
        E::Not(e) => Some((eval(e, t, row)? == 0) as i64),
        E::Bin(l, op, r) => {
            let (a, b) = (eval(l, t, row)?, eval(r, t, row)?);
            Some(match op {
                0 => a.wrapping_add(b),
                1 => a.wrapping_sub(b),
                2 => a.wrapping_mul(b),
                3 => (a > b) as i64,
                4 => (a < b) as i64,
                5 => (a == b) as i64,
                6 => (a != 0 && b != 0) as i64,
                7 => (a != 0 || b != 0) as i64,
                8 if b == 0 => return None,
                8 => a.wrapping_div(b),
                9 => (a >= b) as i64,
                10 => (a <= b) as i64,
                _ => (a != b) as i64,
            })
        }
    }
}

fn cmp_cell(a: Cell, b: Cell, desc: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (None, None) => Equal,
        (None, Some(_)) => Greater, // nulls last in both directions
        (Some(_), None) => Less,
        (Some(x), Some(y)) => {
            if desc {
                y.cmp(&x)
            } else {
                x.cmp(&y)
            }
        }
    }
}

fn pick_distinct(r: &mut Rng, names: &[String], k: usize) -> Vec<String> {
    let mut pool = names.to_vec();
    let mut out = Vec::new();
    while out.len() < k && !pool.is_empty() {
        out.push(pool.remove(r.below(pool.len())));
    }
    out
}

/// Build the plan and the reference result together from the seed.
fn build(seed: u64, left: &Tab, right: &Tab, steps: usize) -> (LogicalPlan, Tab) {
    let mut r = Rng(seed | 1);
    let mut plan = LogicalPlan::scan(left.to_frame());
    let mut t = left.clone();
    let mut joined = false;
    for _ in 0..steps {
        match r.below(7) {
            0 | 1 => {
                let p = gen_bool(&mut r, &t.names, 2);
                plan = plan.filter(to_expr(&p));
                let keep: Vec<bool> = t
                    .rows
                    .iter()
                    .map(|row| eval(&p, &t, row) == Some(1))
                    .collect();
                let mut it = keep.into_iter();
                t.rows.retain(|_| it.next().unwrap_or(false));
            }
            2 => {
                let e = gen_int(&mut r, &t.names, 2);
                let name = ["d", "e", "a", "b", "z"][r.below(5)];
                plan = plan.with_columns(vec![to_expr(&e).alias(name)]);
                let vals: Vec<Cell> = t.rows.iter().map(|row| eval(&e, &t, row)).collect();
                if t.has(name) {
                    let i = t.idx(name);
                    for (row, v) in t.rows.iter_mut().zip(vals) {
                        row[i] = v;
                    }
                } else {
                    t.names.push(name.to_owned());
                    for (row, v) in t.rows.iter_mut().zip(vals) {
                        row.push(v);
                    }
                }
            }
            3 => {
                let k = 1 + r.below(t.names.len());
                let chosen = pick_distinct(&mut r, &t.names, k);
                let refs: Vec<&str> = chosen.iter().map(String::as_str).collect();
                plan = plan.select(&refs);
                let idx: Vec<usize> = chosen.iter().map(|c| t.idx(c)).collect();
                t.rows = t
                    .rows
                    .iter()
                    .map(|row| idx.iter().map(|&i| row[i]).collect())
                    .collect();
                t.names = chosen;
            }
            4 if !joined && t.has("a") => {
                joined = true;
                let how = if r.below(2) == 0 {
                    JoinType::Inner
                } else {
                    JoinType::Left
                };
                plan = plan.join(LogicalPlan::scan(right.to_frame()), &["a"], how);
                let ka = t.idx("a");
                let rk = right.idx("a");
                let right_cols: Vec<usize> = (0..right.names.len()).filter(|&i| i != rk).collect();
                let mut rows = Vec::new();
                for lrow in &t.rows {
                    let mut hit = false;
                    for rrow in &right.rows {
                        if lrow[ka].is_some() && lrow[ka] == rrow[rk] {
                            hit = true;
                            let mut o = lrow.clone();
                            o.extend(right_cols.iter().map(|&i| rrow[i]));
                            rows.push(o);
                        }
                    }
                    if !hit && how == JoinType::Left {
                        let mut o = lrow.clone();
                        o.extend(right_cols.iter().map(|_| None));
                        rows.push(o);
                    }
                }
                for &i in &right_cols {
                    let n = &right.names[i];
                    t.names.push(if t.names.contains(n) {
                        format!("{n}_right")
                    } else {
                        n.clone()
                    });
                }
                t.rows = rows;
            }
            5 => {
                let k = 1 + r.below(2.min(t.names.len()));
                let by = pick_distinct(&mut r, &t.names, k);
                let desc: Vec<bool> = by.iter().map(|_| r.below(2) == 0).collect();
                let refs: Vec<&str> = by.iter().map(String::as_str).collect();
                plan = plan.sort(&refs, &desc);
                let idx: Vec<usize> = by.iter().map(|c| t.idx(c)).collect();
                t.rows.sort_by(|x, y| {
                    for (&i, &d) in idx.iter().zip(&desc) {
                        let o = cmp_cell(x[i], y[i], d);
                        if o != std::cmp::Ordering::Equal {
                            return o;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }
            _ => {
                if r.below(2) == 0 {
                    let n = r.below(6);
                    plan = plan.limit(n);
                    t.rows.truncate(n);
                } else {
                    // Group-by, followed by a sort on all keys so the order is canonical.
                    let nk = 1 + r.below(2.min(t.names.len()));
                    let keys = pick_distinct(&mut r, &t.names, nk);
                    let others: Vec<String> = t
                        .names
                        .iter()
                        .filter(|n| !keys.contains(n))
                        .cloned()
                        .collect();
                    let mut aggs: Vec<(String, AggKind)> = Vec::new();
                    if !others.is_empty() {
                        for _ in 0..1 + r.below(2) {
                            let c = others[r.below(others.len())].clone();
                            let kind = [AggKind::Sum, AggKind::Min, AggKind::Max, AggKind::Count]
                                [r.below(4)];
                            if !aggs.contains(&(c.clone(), kind)) {
                                aggs.push((c, kind));
                            }
                        }
                    }
                    let krefs: Vec<&str> = keys.iter().map(String::as_str).collect();
                    let arefs: Vec<(&str, AggKind)> =
                        aggs.iter().map(|(c, k)| (c.as_str(), *k)).collect();
                    plan = plan
                        .group_by(&krefs, &arefs)
                        .sort(&krefs, &vec![false; krefs.len()]);

                    let kidx: Vec<usize> = keys.iter().map(|c| t.idx(c)).collect();
                    let mut groups: Vec<(Vec<Cell>, Vec<usize>)> = Vec::new();
                    for (ri, row) in t.rows.iter().enumerate() {
                        let key: Vec<Cell> = kidx.iter().map(|&i| row[i]).collect();
                        match groups.iter_mut().find(|(k, _)| *k == key) {
                            Some(g) => g.1.push(ri),
                            None => groups.push((key, vec![ri])),
                        }
                    }
                    let mut rows: Vec<Vec<Cell>> = groups
                        .iter()
                        .map(|(key, members)| {
                            let mut o = key.clone();
                            for (c, kind) in &aggs {
                                let ci = t.idx(c);
                                let vals: Vec<i64> =
                                    members.iter().filter_map(|&m| t.rows[m][ci]).collect();
                                o.push(match kind {
                                    AggKind::Count => Some(vals.len() as i64),
                                    _ if vals.is_empty() => None,
                                    AggKind::Sum => {
                                        Some(vals.iter().fold(0i64, |a, &b| a.wrapping_add(b)))
                                    }
                                    AggKind::Min => vals.iter().copied().min(),
                                    _ => vals.iter().copied().max(),
                                });
                            }
                            o
                        })
                        .collect();
                    rows.sort_by(|x, y| {
                        for i in 0..keys.len() {
                            let o = cmp_cell(x[i], y[i], false);
                            if o != std::cmp::Ordering::Equal {
                                return o;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                    let mut names = keys.clone();
                    for (c, k) in &aggs {
                        let s = match k {
                            AggKind::Sum => "sum",
                            AggKind::Min => "min",
                            AggKind::Max => "max",
                            AggKind::Count => "count",
                            AggKind::Mean => "mean",
                        };
                        names.push(format!("{c}_{s}"));
                    }
                    t = Tab { names, rows };
                }
            }
        }
    }
    (plan, t)
}

fn table(names: &[&str], rows: &[Vec<Cell>]) -> Tab {
    Tab {
        names: names.iter().map(|s| (*s).to_owned()).collect(),
        rows: rows.to_vec(),
    }
}

fn cells(width: usize, max_rows: usize) -> impl Strategy<Value = Vec<Vec<Cell>>> {
    prop::collection::vec(
        prop::collection::vec(prop::option::weighted(0.85, -3i64..4), width),
        0..max_rows,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]
    #[test]
    fn whole_query_matches_reference(
        lrows in cells(3, 40),
        rrows in cells(3, 20),
        clash in any::<bool>(),
        seed in any::<u64>(),
        steps in 1usize..7,
    ) {
        let left = table(&["a", "b", "c"], &lrows);
        let right = if clash {
            table(&["a", "b", "z"], &rrows)
        } else {
            table(&["a", "w", "z"], &rrows)
        };
        let (plan, expected) = build(seed, &left, &right, steps);
        let unopt = Tab::from_frame(&plan.collect().unwrap());
        prop_assert_eq!(&unopt, &expected, "unoptimized\n{}", plan);
        let opt = plan.clone().optimize();
        let got = Tab::from_frame(&opt.collect().unwrap());
        prop_assert_eq!(&got, &expected, "optimized\n{}\nfrom\n{}", opt, plan);
    }
}
