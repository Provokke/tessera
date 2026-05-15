//! Hash group-by with sum / mean / min / max / count.
//!
//! Groups are emitted in first-occurrence order. Null keys form their own
//! group. Aggregations skip nulls; `sum`/`mean`/`min`/`max` of a group with
//! no valid values is null, `count` counts non-null values. See
//! [`DataFrame::group_by`] for the execution strategy.
use std::collections::HashMap;

use hashbrown::HashTable;
use rayon::prelude::*;
use tessera_arrow::{Array, PrimitiveArray};

use crate::keys::{check_rows, hash_rows, rows_eq, KeyCol};
use crate::{CoreError, DataFrame, DataType, Result, Scalar, Series};

/// Aggregation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    /// Sum (i64 for integer inputs, f64 for floats).
    Sum,
    /// Mean as f64.
    Mean,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Number of non-null values (i64).
    Count,
}

impl AggKind {
    fn suffix(self) -> &'static str {
        match self {
            AggKind::Sum => "sum",
            AggKind::Mean => "mean",
            AggKind::Min => "min",
            AggKind::Max => "max",
            AggKind::Count => "count",
        }
    }
}

fn groups_from_ids(gid: &[u32], n: usize) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); n];
    for (i, &g) in gid.iter().enumerate() {
        groups[g as usize].push(i);
    }
    groups
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) enum Key {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(String),
}

pub(crate) fn key(s: Scalar) -> Key {
    match s {
        Scalar::Null(_) => Key::Null,
        Scalar::Boolean(b) => Key::Bool(b),
        Scalar::Int32(v) => Key::Int(v as i64),
        Scalar::Int64(v) => Key::Int(v),
        // Normalise -0.0 to 0.0 so equal floats share a group.
        Scalar::Float64(v) => Key::Float(if v == 0.0 { 0 } else { v.to_bits() }),
        Scalar::Utf8(s) => Key::Str(s),
    }
}

enum Num {
    I(i64),
    F(f64),
}

fn num(a: &Array, i: usize) -> Option<Num> {
    match a {
        Array::Int32(p) => p.get(i).map(|v| Num::I(v as i64)),
        Array::Int64(p) => p.get(i).map(Num::I),
        Array::Float64(p) => p.get(i).map(Num::F),
        _ => None,
    }
}

fn aggregate(col: &Series, groups: &[Vec<usize>], kind: AggKind) -> Result<Array> {
    let a = col.array();
    let is_float = match a.data_type() {
        DataType::Float64 => true,
        DataType::Int32 | DataType::Int64 => false,
        _ if kind == AggKind::Count => false,
        _ => {
            return Err(CoreError::ColumnNotFound(format!(
                "{} (not numeric)",
                col.name()
            )))
        }
    };
    let out = match kind {
        AggKind::Count => {
            Array::Int64(PrimitiveArray::from_options(groups.iter().map(|g| {
                Some(g.iter().filter(|&&i| !a.scalar(i).is_null()).count() as i64)
            })))
        }
        AggKind::Mean => Array::Float64(PrimitiveArray::from_options(groups.iter().map(|g| {
            let v: Vec<f64> = g
                .iter()
                .filter_map(|&i| num(a, i))
                .map(|n| match n {
                    Num::I(x) => x as f64,
                    Num::F(x) => x,
                })
                .collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
        }))),
        _ if is_float => Array::Float64(PrimitiveArray::from_options(groups.iter().map(|g| {
            let v = g.iter().filter_map(|&i| match num(a, i) {
                Some(Num::F(x)) => Some(x),
                _ => None,
            });
            match kind {
                AggKind::Sum => v.reduce(|x, y| x + y),
                AggKind::Min => v.filter(|x| !x.is_nan()).reduce(f64::min),
                _ => v.filter(|x| !x.is_nan()).reduce(f64::max),
            }
        }))),
        _ => Array::Int64(PrimitiveArray::from_options(groups.iter().map(|g| {
            let v = g.iter().filter_map(|&i| match num(a, i) {
                Some(Num::I(x)) => Some(x),
                _ => None,
            });
            match kind {
                AggKind::Sum => v.reduce(i64::wrapping_add),
                AggKind::Min => v.min(),
                _ => v.max(),
            }
        }))),
    };
    Ok(out)
}

/// Column-wise accumulators for one aggregation in the integer-key fast path.
fn agg_columnwise(a: &Array, gid: &[u32], ngroups: usize, kind: AggKind) -> Option<Array> {
    let valid = |i: usize| match a {
        Array::Int64(p) => p.is_valid(i),
        Array::Float64(p) => p.is_valid(i),
        Array::Int32(p) => p.is_valid(i),
        _ => false,
    };
    let mut count = vec![0i64; ngroups];
    for (i, &g) in gid.iter().enumerate() {
        count[g as usize] += valid(i) as i64;
    }
    let present = |g: usize| count[g] > 0;
    Some(match (kind, a) {
        (AggKind::Count, _) => {
            Array::Int64(PrimitiveArray::from_options(count.iter().map(|c| Some(*c))))
        }
        (AggKind::Mean, Array::Int64(_) | Array::Float64(_) | Array::Int32(_)) => {
            let mut sum = vec![0.0f64; ngroups];
            for (i, &g) in gid.iter().enumerate() {
                if valid(i) {
                    sum[g as usize] += match a {
                        Array::Int64(p) => p.values()[i] as f64,
                        Array::Int32(p) => p.values()[i] as f64,
                        Array::Float64(p) => p.values()[i],
                        _ => 0.0,
                    };
                }
            }
            Array::Float64(PrimitiveArray::from_options(
                (0..ngroups).map(|g| present(g).then(|| sum[g] / count[g] as f64)),
            ))
        }
        (_, Array::Int64(p)) => {
            let init = match kind {
                AggKind::Min => i64::MAX,
                AggKind::Max => i64::MIN,
                _ => 0,
            };
            let mut acc = vec![init; ngroups];
            let v = p.values();
            for (i, &g) in gid.iter().enumerate() {
                if valid(i) {
                    let s = &mut acc[g as usize];
                    *s = match kind {
                        AggKind::Min => (*s).min(v[i]),
                        AggKind::Max => (*s).max(v[i]),
                        _ => s.wrapping_add(v[i]),
                    };
                }
            }
            Array::Int64(PrimitiveArray::from_options(
                (0..ngroups).map(|g| present(g).then_some(acc[g])),
            ))
        }
        (_, Array::Float64(p)) => {
            let v = p.values();
            let mut acc: Vec<Option<f64>> = vec![None; ngroups];
            for (i, &g) in gid.iter().enumerate() {
                if !valid(i) {
                    continue;
                }
                let x = v[i];
                let s = &mut acc[g as usize];
                *s = match kind {
                    AggKind::Sum => Some(s.map_or(x, |a| a + x)),
                    // NaN is skipped, matching the row-wise path.
                    _ if x.is_nan() => *s,
                    AggKind::Min => Some(s.map_or(x, |a| a.min(x))),
                    _ => Some(s.map_or(x, |a| a.max(x))),
                };
            }
            Array::Float64(PrimitiveArray::from_options(acc))
        }
        _ => return None,
    })
}

impl DataFrame {
    /// Fast path: single `Int64` key column. Rows are mapped to dense group
    /// ids with a `hashbrown` map keyed on `Option<i64>` (no per-row `Scalar`
    /// or `Vec` allocation), then each aggregation is a single column pass.
    fn group_by_i64(
        &self,
        kc: &Series,
        k: &PrimitiveArray<i64>,
        aggs: &[(&str, AggKind)],
    ) -> Result<DataFrame> {
        let mut map: hashbrown::HashMap<Option<i64>, u32> = hashbrown::HashMap::new();
        let mut firsts: Vec<usize> = Vec::new();
        let vals = k.values();
        let gid: Vec<u32> = (0..k.len())
            .map(|i| {
                let key = k.is_valid(i).then(|| vals[i]);
                *map.entry(key).or_insert_with(|| {
                    firsts.push(i);
                    (firsts.len() - 1) as u32
                })
            })
            .collect();
        let mut out = vec![kc.take(&firsts)?];
        for (name, kind) in aggs {
            let col = self.column(name)?;
            let arr = match agg_columnwise(col.array(), &gid, firsts.len(), *kind) {
                Some(a) => a,
                None => {
                    // Non-numeric input: reuse the generic path for its error/semantics.
                    let groups = groups_from_ids(&gid, firsts.len());
                    aggregate(col, &groups, *kind)?
                }
            };
            out.push(Series::new(format!("{name}_{}", kind.suffix()), arr));
        }
        DataFrame::new(out)
    }

    /// Group by `keys` and compute `aggs` as `(column, kind)` pairs.
    /// Output: key columns, then one column per agg named `{col}_{kind}`.
    ///
    /// A single `Int64` key uses a column-wise fast path. Everything else
    /// uses the parallel partitioned hash group-by: per-row `u64` hashes
    /// ([`tessera_compute::hash`]) split the rows into partitions, each
    /// partition builds its own `hashbrown::HashTable` on a rayon task, and the
    /// groups are renumbered by first occurrence.
    pub fn group_by(&self, keys: &[&str], aggs: &[(&str, AggKind)]) -> Result<DataFrame> {
        let key_cols = self.key_columns(keys)?;
        if let [kc] = key_cols.as_slice() {
            if let Array::Int64(k) = kc.array() {
                return self.group_by_i64(kc, k, aggs);
            }
        }
        let parts = if self.height() >= PARALLEL_MIN_ROWS {
            rayon::current_num_threads().next_power_of_two().min(64)
        } else {
            1
        };
        self.group_by_partitioned(&key_cols, aggs, parts)
    }

    /// Like [`DataFrame::group_by`] but always takes the partitioned hash path
    /// (never the `Int64` fast path) with an explicit partition count. Exposed
    /// so tests can compare partition counts against the reference.
    pub fn group_by_with_partitions(
        &self,
        keys: &[&str],
        aggs: &[(&str, AggKind)],
        partitions: usize,
    ) -> Result<DataFrame> {
        let key_cols = self.key_columns(keys)?;
        self.group_by_partitioned(&key_cols, aggs, partitions.max(1))
    }

    fn key_columns(&self, keys: &[&str]) -> Result<Vec<&Series>> {
        keys.iter().map(|k| self.column(k)).collect()
    }

    fn group_by_partitioned(
        &self,
        key_cols: &[&Series],
        aggs: &[(&str, AggKind)],
        parts: usize,
    ) -> Result<DataFrame> {
        let arrays: Vec<&Array> = key_cols.iter().map(|c| c.array()).collect();
        let (gid, firsts) = assign_groups(&arrays, self.height(), parts)?;
        let ngroups = firsts.len();
        let first_idx: Vec<Option<u32>> = firsts.iter().map(|&r| Some(r)).collect();
        let mut out = key_cols
            .iter()
            .map(|c| c.take_opt(&first_idx))
            .collect::<Result<Vec<_>>>()?;
        let agg_cols = aggs
            .par_iter()
            .map(|(name, kind)| {
                let col = self.column(name)?;
                let arr = match agg_columnwise(col.array(), &gid, ngroups, *kind) {
                    Some(a) => a,
                    None => aggregate(col, &groups_from_ids(&gid, ngroups), *kind)?,
                };
                Ok(Series::new(format!("{name}_{}", kind.suffix()), arr))
            })
            .collect::<Result<Vec<_>>>()?;
        out.extend(agg_cols);
        DataFrame::new(out)
    }

    /// Reference implementation: single-threaded, std `HashMap` over
    /// materialized `Vec<Key>` rows. Used to test the optimised paths.
    pub fn group_by_reference(&self, keys: &[&str], aggs: &[(&str, AggKind)]) -> Result<DataFrame> {
        let key_cols = self.key_columns(keys)?;
        let mut index: HashMap<Vec<Key>, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for r in 0..self.height() {
            let k: Vec<Key> = key_cols.iter().map(|c| key(c.get(r))).collect();
            let g = *index.entry(k).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(r);
        }
        let firsts: Vec<usize> = groups.iter().filter_map(|g| g.first().copied()).collect();
        let mut out = key_cols
            .iter()
            .map(|c| c.take(&firsts))
            .collect::<Result<Vec<_>>>()?;
        for (name, kind) in aggs {
            let col = self.column(name)?;
            out.push(Series::new(
                format!("{name}_{}", kind.suffix()),
                aggregate(col, &groups, *kind)?,
            ));
        }
        DataFrame::new(out)
    }
}

/// Below this many rows the partitioned path uses one partition.
const PARALLEL_MIN_ROWS: usize = 1 << 15;

/// Group id per row (dense, numbered by first occurrence) and the first row of
/// each group.
fn assign_groups(keys: &[&Array], n: usize, parts: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    check_rows(n)?;
    let hashes = hash_rows(keys, n)?;
    let cols: Vec<KeyCol<'_>> = keys.iter().map(|a| KeyCol::new(a)).collect();
    // Partition on hash bits 32.. so the low bits (bucket index) and the top
    // 7 bits (control bytes) stay well mixed inside each table.
    let part_of = |h: u64| ((h >> 32) as usize) % parts;
    let mut prows: Vec<Vec<u32>> = (0..parts)
        .map(|_| Vec::with_capacity(n / parts + 16))
        .collect();
    for (r, &h) in hashes.iter().enumerate() {
        prows[part_of(h)].push(r as u32);
    }
    // Per partition: local group id per row position, and first row per local group.
    let per_part: Vec<(Vec<u32>, Vec<u32>)> = prows
        .par_iter()
        .map(|rows| {
            let mut table: HashTable<u32> = HashTable::new();
            let mut firsts: Vec<u32> = Vec::new();
            let mut local = Vec::with_capacity(rows.len());
            for &r in rows {
                let h = hashes[r as usize];
                let found = table
                    .find(h, |&g| {
                        rows_eq(&cols, r as usize, firsts[g as usize] as usize)
                    })
                    .copied();
                let g = match found {
                    Some(g) => g,
                    None => {
                        let g = firsts.len() as u32;
                        firsts.push(r);
                        table.insert_unique(h, g, |&g| hashes[firsts[g as usize] as usize]);
                        g
                    }
                };
                local.push(g);
            }
            (local, firsts)
        })
        .collect();
    // Renumber groups globally by first-occurrence row.
    let mut order: Vec<(u32, usize, u32)> = per_part
        .iter()
        .enumerate()
        .flat_map(|(p, (_, firsts))| {
            firsts
                .iter()
                .enumerate()
                .map(move |(g, &r)| (r, p, g as u32))
        })
        .collect();
    order.sort_unstable_by_key(|&(r, _, _)| r);
    let mut remap: Vec<Vec<u32>> = per_part.iter().map(|(_, f)| vec![0; f.len()]).collect();
    for (gi, &(_, p, g)) in order.iter().enumerate() {
        remap[p][g as usize] = gi as u32;
    }
    let mut gid = vec![0u32; n];
    for (p, rows) in prows.iter().enumerate() {
        for (&r, &g) in rows.iter().zip(&per_part[p].0) {
            gid[r as usize] = remap[p][g as usize];
        }
    }
    Ok((gid, order.into_iter().map(|(r, _, _)| r).collect()))
}
