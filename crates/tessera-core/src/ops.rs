//! Sorting (single and multi-key) and hash joins (inner, left, right and full
//! outer; multi-column keys, optionally differently named on each side).
use std::cmp::Ordering;

use hashbrown::HashTable;
use rayon::prelude::*;
use tessera_arrow::{Array, DataType, PrimitiveArray};
use tessera_compute::sort::{argsort_f64, argsort_i32, argsort_i64, argsort_utf8};

use crate::keys::{check_rows, hash_rows, rows_eq, KeyCol};
use crate::{CoreError, DataFrame, Result, Scalar, Series};

fn cmp_non_null(a: &Scalar, b: &Scalar) -> Ordering {
    match (a, b) {
        (Scalar::Boolean(x), Scalar::Boolean(y)) => x.cmp(y),
        (Scalar::Int32(x), Scalar::Int32(y)) => x.cmp(y),
        (Scalar::Int64(x), Scalar::Int64(y)) => x.cmp(y),
        (Scalar::Float64(x), Scalar::Float64(y)) => x.total_cmp(y),
        (Scalar::Utf8(x), Scalar::Utf8(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// Stable LSD radix argsort (8 passes of 8 bits, constant-byte passes
/// skipped) for i64; nulls are appended last in original order. Descending
/// order sorts the bitwise complement of the key, which keeps ties stable.
pub fn radix_argsort_i64(a: &PrimitiveArray<i64>, descending: bool) -> Vec<usize> {
    let v = a.values();
    let mut keys: Vec<(u64, usize)> = Vec::with_capacity(a.len());
    let mut nulls = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if a.is_valid(i) {
            let k = (x as u64) ^ (1 << 63);
            keys.push((if descending { !k } else { k }, i));
        } else {
            nulls.push(i);
        }
    }
    let mut tmp = vec![(0u64, 0usize); keys.len()];
    for pass in 0..8 {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        for (k, _) in &keys {
            counts[((k >> shift) & 0xFF) as usize] += 1;
        }
        if counts.contains(&keys.len()) {
            continue;
        }
        let mut pos = [0usize; 256];
        let mut acc = 0;
        for (p, c) in pos.iter_mut().zip(counts.iter()) {
            *p = acc;
            acc += c;
        }
        for &(k, i) in &keys {
            let b = ((k >> shift) & 0xFF) as usize;
            tmp[pos[b]] = (k, i);
            pos[b] += 1;
        }
        std::mem::swap(&mut keys, &mut tmp);
    }
    keys.into_iter().map(|(_, i)| i).chain(nulls).collect()
}

/// Join type for [`DataFrame::join_on`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Only rows with a match on both sides.
    Inner,
    /// Every left row; unmatched rows get nulls in the right columns.
    Left,
    /// Every right row; unmatched rows get nulls in the left columns.
    Right,
    /// Every row of both sides (full outer join).
    Outer,
}

/// Stable argsort of any array (nulls last); booleans sort false < true.
fn argsort_array(a: &Array, descending: bool) -> Result<Vec<u32>> {
    Ok(match a {
        Array::Int32(x) => argsort_i32(x, descending)?,
        Array::Int64(x) => argsort_i64(x, descending)?,
        Array::Float64(x) => argsort_f64(x, descending)?,
        Array::Utf8(x) => argsort_utf8(x, descending)?,
        Array::Boolean(x) => {
            let as_int = PrimitiveArray::from_options(x.iter().map(|v| v.map(i32::from)));
            argsort_i32(&as_int, descending)?
        }
    })
}

/// Type of a same-named key column after a right or outer join fills the left
/// key from the right one: equal types stay, two integer widths widen to
/// `Int64`, anything else is not mergeable.
pub fn merged_key_type(l: DataType, r: DataType) -> Option<DataType> {
    match (l, r) {
        (a, b) if a == b => Some(a),
        (DataType::Int32 | DataType::Int64, DataType::Int32 | DataType::Int64) => {
            Some(DataType::Int64)
        }
        _ => None,
    }
}

/// Left key values, with right-only rows (`li[k] == None`) taken from the
/// right key. `None` when the key types cannot be merged into one column.
fn coalesce_key(l: &Series, r: &Series, li: &[Option<u32>], ri: &[Option<u32>]) -> Option<Series> {
    let name = l.name();
    let pick = |k: usize| match (li[k], ri[k]) {
        (Some(i), _) => l.get(i as usize),
        (None, Some(j)) => r.get(j as usize),
        (None, None) => Scalar::Null(l.data_type()),
    };
    let int = |s: Scalar| match s {
        Scalar::Int32(v) => Some(i64::from(v)),
        Scalar::Int64(v) => Some(v),
        _ => None,
    };
    let n = li.len();
    let arr = match (l.array(), r.array()) {
        (Array::Int32(_), Array::Int32(_)) => Array::Int32(PrimitiveArray::from_options(
            (0..n).map(|k| int(pick(k)).and_then(|v| i32::try_from(v).ok())),
        )),
        (Array::Int32(_) | Array::Int64(_), Array::Int32(_) | Array::Int64(_)) => {
            Array::Int64(PrimitiveArray::from_options((0..n).map(|k| int(pick(k)))))
        }
        (Array::Float64(_), Array::Float64(_)) => Array::Float64(PrimitiveArray::from_options(
            (0..n).map(|k| pick(k).as_f64()),
        )),
        (Array::Boolean(_), Array::Boolean(_)) => Array::Boolean(
            tessera_arrow::BooleanArray::from_options((0..n).map(|k| match pick(k) {
                Scalar::Boolean(b) => Some(b),
                _ => None,
            })),
        ),
        (Array::Utf8(_), Array::Utf8(_)) => {
            let vals: Vec<Option<String>> = (0..n)
                .map(|k| match pick(k) {
                    Scalar::Utf8(s) => Some(s),
                    _ => None,
                })
                .collect();
            Array::Utf8(
                tessera_arrow::Utf8Array::from_options(vals.iter().map(|v| v.as_deref())).ok()?,
            )
        }
        _ => return None,
    };
    Some(Series::new(name, arr))
}

/// One side of a probe: the probe key columns and the build-side table.
struct BuildSide<'a> {
    keys: Vec<KeyCol<'a>>,
    table: HashTable<u32>,
    /// First row of each key group (index = group id).
    head: Vec<u32>,
    /// Next row with the same key, `u32::MAX` at the end of a chain.
    next: Vec<u32>,
}

/// Matched (left, right) row indices; a `None` right index is an unmatched left row.
type IndexPair = (Vec<Option<u32>>, Vec<Option<u32>>);

const END: u32 = u32::MAX;
const PROBE_CHUNK: usize = 1 << 15;

impl DataFrame {
    /// Stable sort by one column; nulls always last. Floats use IEEE total order.
    pub fn sort_by(&self, column: &str, descending: bool) -> Result<DataFrame> {
        self.sort_by_multi(&[column], &[descending])
    }

    /// Stable sort by several columns (first = most significant), each with its
    /// own direction; nulls are last in every key. Implemented as least
    /// significant key first stable argsorts using the `tessera_compute::sort`
    /// radix/string kernels, then one gather per column.
    pub fn sort_by_multi(&self, by: &[&str], descending: &[bool]) -> Result<DataFrame> {
        if by.len() != descending.len() {
            return Err(CoreError::Invalid(format!(
                "{} sort keys but {} directions",
                by.len(),
                descending.len()
            )));
        }
        let n = self.height();
        check_rows(n)?;
        let keys = by
            .iter()
            .map(|k| self.column(k))
            .collect::<Result<Vec<_>>>()?;
        if keys.is_empty() {
            return Ok(self.clone());
        }
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for (i, (col, &desc)) in keys.iter().zip(descending).enumerate().rev() {
            let order = if i == keys.len() - 1 {
                argsort_array(col.array(), desc)?
            } else {
                let idx: Vec<Option<u32>> = perm.iter().map(|&p| Some(p)).collect();
                argsort_array(col.take_opt(&idx)?.array(), desc)?
            };
            perm = order.iter().map(|&o| perm[o as usize]).collect();
        }
        self.gather(&perm.iter().map(|&p| Some(p)).collect::<Vec<_>>())
    }

    fn gather(&self, idx: &[Option<u32>]) -> Result<DataFrame> {
        let cols = self
            .columns()
            .par_iter()
            .map(|c| c.take_opt(idx))
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(cols)
    }

    /// Reference sort: stable comparison sort over `Scalar`s (any type).
    pub fn sort_by_comparison(&self, column: &str, descending: bool) -> Result<DataFrame> {
        self.sort_by_multi_comparison(&[column], &[descending])
    }

    /// Reference multi-key sort: one stable comparison sort over `Scalar`s.
    pub fn sort_by_multi_comparison(&self, by: &[&str], descending: &[bool]) -> Result<DataFrame> {
        if by.len() != descending.len() {
            return Err(CoreError::Invalid(format!(
                "{} sort keys but {} directions",
                by.len(),
                descending.len()
            )));
        }
        let keys = by
            .iter()
            .map(|k| self.column(k))
            .collect::<Result<Vec<_>>>()?;
        let vals: Vec<Vec<Scalar>> = keys
            .iter()
            .map(|c| (0..self.height()).map(|i| c.get(i)).collect())
            .collect();
        let mut idx: Vec<usize> = (0..self.height()).collect();
        idx.sort_by(|&a, &b| {
            for (v, &desc) in vals.iter().zip(descending) {
                let o = match (v[a].is_null(), v[b].is_null()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => {
                        let o = cmp_non_null(&v[a], &v[b]);
                        if desc {
                            o.reverse()
                        } else {
                            o
                        }
                    }
                };
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        });
        let cols = self
            .columns()
            .iter()
            .map(|c| c.take(&idx))
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(cols)
    }

    /// Inner hash join on a column present in both frames (see
    /// [`DataFrame::join_on`]).
    pub fn join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        self.join_on(other, &[on], JoinType::Inner)
    }

    /// Hash join on one or more same-named key columns; see
    /// [`DataFrame::join_keys`].
    pub fn join_on(&self, other: &DataFrame, on: &[&str], how: JoinType) -> Result<DataFrame> {
        self.join_keys(other, on, on, how)
    }

    /// Hash join of `self.left_on[i] == other.right_on[i]` (all pairs must
    /// match). Builds a hash table on `other`, probes with `self` in parallel
    /// row chunks. Output rows: first the probe results in left-row order, then
    /// right-row order (a left row without a match is kept for `Left` and
    /// `Outer`, with null right columns), then for `Right` and `Outer` every
    /// unmatched right row in right-row order (null left columns). Rows with a
    /// null in any key never match (SQL semantics), so they are only ever kept
    /// as unmatched rows. Output columns: all left columns, then the right
    /// columns. A right key column is dropped when it has the same name as its
    /// left key (for `Right`/`Outer` the left key column is then filled from
    /// the right key on right-only rows, so the two key types must be
    /// compatible, see [`merged_key_type`], else the join is an error); a right
    /// key with a different name is kept. A right name clashing with a left column gets a `_right`
    /// suffix. Integer key widths compare by value; other type mismatches never
    /// match.
    pub fn join_keys(
        &self,
        other: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        how: JoinType,
    ) -> Result<DataFrame> {
        if left_on.is_empty() {
            return Err(CoreError::Invalid("join needs at least one key".into()));
        }
        if left_on.len() != right_on.len() {
            return Err(CoreError::Invalid(format!(
                "{} left keys but {} right keys",
                left_on.len(),
                right_on.len()
            )));
        }
        check_rows(self.height())?;
        check_rows(other.height())?;
        let lk: Vec<&Array> = left_on
            .iter()
            .map(|k| self.column(k).map(Series::array))
            .collect::<Result<_>>()?;
        let rk: Vec<&Array> = right_on
            .iter()
            .map(|k| other.column(k).map(Series::array))
            .collect::<Result<_>>()?;

        // Build on the right.
        let rh = hash_rows(&rk, other.height())?;
        let mut side = BuildSide {
            keys: rk.iter().map(|a| KeyCol::new(a)).collect(),
            table: HashTable::new(),
            head: Vec::new(),
            next: vec![END; other.height()],
        };
        let mut tail: Vec<u32> = Vec::new();
        for r in 0..other.height() {
            if side.keys.iter().any(|k| k.is_null(r)) {
                continue;
            }
            let found = side
                .table
                .find(rh[r], |&g| {
                    rows_eq(&side.keys, r, side.head[g as usize] as usize)
                })
                .copied();
            match found {
                Some(g) => {
                    side.next[tail[g as usize] as usize] = r as u32;
                    tail[g as usize] = r as u32;
                }
                None => {
                    let g = side.head.len() as u32;
                    side.head.push(r as u32);
                    tail.push(r as u32);
                    let head = &side.head;
                    side.table
                        .insert_unique(rh[r], g, |&g| rh[head[g as usize] as usize]);
                }
            }
        }

        // Probe with the left in parallel chunks, preserving left-row order.
        let lh = hash_rows(&lk, self.height())?;
        let lkeys: Vec<KeyCol<'_>> = lk.iter().map(|a| KeyCol::new(a)).collect();
        let starts: Vec<usize> = (0..self.height()).step_by(PROBE_CHUNK).collect();
        let parts: Vec<IndexPair> = starts
            .par_iter()
            .map(|&s| {
                let end = (s + PROBE_CHUNK).min(self.height());
                let (mut li, mut ri) = (Vec::new(), Vec::new());
                for (l, &h) in lh.iter().enumerate().take(end).skip(s) {
                    let mut matched = false;
                    if !lkeys.iter().any(|k| k.is_null(l)) {
                        let found = side.table.find(h, |&g| {
                            let h = side.head[g as usize] as usize;
                            lkeys.iter().zip(&side.keys).all(|(a, b)| a.eq(l, b, h))
                        });
                        if let Some(&g) = found {
                            let mut r = side.head[g as usize];
                            while r != END {
                                li.push(Some(l as u32));
                                ri.push(Some(r));
                                r = side.next[r as usize];
                            }
                            matched = true;
                        }
                    }
                    if !matched && matches!(how, JoinType::Left | JoinType::Outer) {
                        li.push(Some(l as u32));
                        ri.push(None);
                    }
                }
                (li, ri)
            })
            .collect();
        let total: usize = parts.iter().map(|p| p.0.len()).sum();
        let (mut li, mut ri) = (Vec::with_capacity(total), Vec::with_capacity(total));
        for (a, b) in parts {
            li.extend(a);
            ri.extend(b);
        }
        let keep_unmatched_right = matches!(how, JoinType::Right | JoinType::Outer);
        if keep_unmatched_right {
            let mut seen = vec![false; other.height()];
            for r in ri.iter().flatten() {
                seen[*r as usize] = true;
            }
            for (r, _) in seen.iter().enumerate().filter(|(_, s)| !**s) {
                li.push(None);
                ri.push(Some(r as u32));
            }
        }

        let mut cols = self.gather(&li)?.columns().to_vec();
        // Same-named key pairs: the right column is dropped, and with unmatched
        // right rows the left column is filled from it where the types allow.
        let mut drop_right: Vec<&str> = Vec::new();
        for (l, r) in left_on.iter().zip(right_on) {
            if l != r {
                continue;
            }
            if !keep_unmatched_right {
                drop_right.push(r);
            } else {
                let merged =
                    coalesce_key(self.column(l)?, other.column(r)?, &li, &ri).ok_or_else(|| {
                        CoreError::Invalid(format!(
                            "right/outer join key {l:?} has incompatible types on the two sides"
                        ))
                    })?;
                if let Some(slot) = cols.iter_mut().find(|c| c.name() == *l) {
                    *slot = merged;
                }
                drop_right.push(r);
            }
        }
        let right_cols: Vec<&Series> = other
            .columns()
            .iter()
            .filter(|c| !drop_right.contains(&c.name()))
            .collect();
        let taken = right_cols
            .par_iter()
            .map(|c| c.take_opt(&ri))
            .collect::<Result<Vec<_>>>()?;
        for t in taken {
            let name = if self.column(t.name()).is_ok() {
                format!("{}_right", t.name())
            } else {
                t.name().to_owned()
            };
            cols.push(Series::new(name, t.array().clone()));
        }
        DataFrame::new(cols)
    }
}
