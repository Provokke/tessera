//! Logical plan, schema inference and execution.
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessera_arrow::Array;
use tessera_core::{merged_key_type, AggKind, CoreError, DataFrame, DataType, JoinType, Series};
use tessera_io::{read_csv, read_csv_header, CsvOptions};

use crate::expr::Expr;

/// Column names with (when known) their types.
pub type Schema = Vec<(String, DataType)>;

/// A logical plan.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// In-memory table scan. `predicate` and `projection` are filled in by the
    /// optimiser (the predicate is applied before the projection, so it may
    /// use columns the projection drops).
    Scan {
        /// The table.
        df: Arc<DataFrame>,
        /// Columns to keep, in this order (`None` = all).
        projection: Option<Vec<String>>,
        /// Boolean predicate applied while scanning.
        predicate: Option<Expr>,
    },
    /// CSV file scan; a projection is pushed into the parser.
    ScanCsv {
        /// File path.
        path: PathBuf,
        /// Parser options (the optimiser sets `projection`).
        options: CsvOptions,
        /// Header of the file, in file order.
        header: Vec<String>,
    },
    /// Keep rows where the predicate is true.
    Filter {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// Boolean predicate.
        predicate: Expr,
    },
    /// Project columns by name.
    Select {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// Column names.
        columns: Vec<String>,
    },
    /// Add or replace columns, applied in order (later expressions see earlier ones).
    WithColumns {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// `(output name, expression)` pairs.
        exprs: Vec<(String, Expr)>,
    },
    /// Group by key columns and aggregate (output: keys, then `{col}_{agg}`).
    GroupBy {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// Key columns.
        keys: Vec<String>,
        /// `(column, aggregation)` pairs.
        aggs: Vec<(String, AggKind)>,
    },
    /// Hash join of `left_on[i] == right_on[i]` (same or different names).
    Join {
        /// Left (probe) side.
        left: Box<LogicalPlan>,
        /// Right (build) side.
        right: Box<LogicalPlan>,
        /// Key columns of the left side.
        left_on: Vec<String>,
        /// Key columns of the right side, paired with `left_on`.
        right_on: Vec<String>,
        /// Inner, left, right or full outer.
        how: JoinType,
    },
    /// Stable multi-key sort (nulls last per key).
    Sort {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// Sort keys.
        by: Vec<String>,
        /// Per-key descending flags.
        descending: Vec<bool>,
    },
    /// First `n` rows.
    Limit {
        /// Input plan.
        input: Box<LogicalPlan>,
        /// Row count.
        n: usize,
    },
}

pub(crate) fn agg_suffix(kind: AggKind) -> &'static str {
    match kind {
        AggKind::Sum => "sum",
        AggKind::Mean => "mean",
        AggKind::Min => "min",
        AggKind::Max => "max",
        AggKind::Count => "count",
    }
}

fn csv_err(e: impl fmt::Display) -> CoreError {
    CoreError::Invalid(format!("csv: {e}"))
}

fn find<'a>(
    cols: &'a [(String, Option<DataType>)],
    name: &str,
) -> Result<&'a Option<DataType>, CoreError> {
    cols.iter()
        .find(|(n, _)| n == name)
        .map(|(_, d)| d)
        .ok_or_else(|| CoreError::ColumnNotFound(name.to_owned()))
}

impl LogicalPlan {
    /// Scan a frame.
    pub fn scan(df: DataFrame) -> Self {
        LogicalPlan::Scan {
            df: Arc::new(df),
            projection: None,
            predicate: None,
        }
    }
    /// Scan a CSV file. Only the header is read here; the optimiser pushes
    /// the set of needed columns into the parser.
    pub fn scan_csv(path: impl AsRef<Path>, options: CsvOptions) -> Result<Self, CoreError> {
        let header = read_csv_header(path.as_ref()).map_err(csv_err)?;
        Ok(LogicalPlan::ScanCsv {
            path: path.as_ref().to_path_buf(),
            options,
            header,
        })
    }
    /// Add a filter.
    pub fn filter(self, predicate: Expr) -> Self {
        LogicalPlan::Filter {
            input: Box::new(self),
            predicate,
        }
    }
    /// Add a projection.
    pub fn select(self, columns: &[&str]) -> Self {
        LogicalPlan::Select {
            input: Box::new(self),
            columns: columns.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
    /// Add or replace columns. Each expression is named by its alias (or the
    /// column name of a bare column, or its printed form).
    pub fn with_columns(self, exprs: Vec<Expr>) -> Self {
        let exprs = exprs
            .into_iter()
            .map(|e| {
                let name = e.output_name().map_or_else(|| e.to_string(), str::to_owned);
                (name, e.unaliased().clone())
            })
            .collect();
        LogicalPlan::WithColumns {
            input: Box::new(self),
            exprs,
        }
    }
    /// Group by `keys` and aggregate.
    pub fn group_by(self, keys: &[&str], aggs: &[(&str, AggKind)]) -> Self {
        LogicalPlan::GroupBy {
            input: Box::new(self),
            keys: keys.iter().map(|s| (*s).to_owned()).collect(),
            aggs: aggs.iter().map(|(c, k)| ((*c).to_owned(), *k)).collect(),
        }
    }
    /// Join with another plan on same-named key columns.
    pub fn join(self, right: LogicalPlan, on: &[&str], how: JoinType) -> Self {
        self.join_with(right, on, on, how)
    }
    /// Join with another plan on `left_on[i] == right_on[i]` (see
    /// [`tessera_core::DataFrame::join_keys`] for the output columns).
    pub fn join_with(
        self,
        right: LogicalPlan,
        left_on: &[&str],
        right_on: &[&str],
        how: JoinType,
    ) -> Self {
        LogicalPlan::Join {
            left: Box::new(self),
            right: Box::new(right),
            left_on: left_on.iter().map(|s| (*s).to_owned()).collect(),
            right_on: right_on.iter().map(|s| (*s).to_owned()).collect(),
            how,
        }
    }
    /// Sort by columns.
    pub fn sort(self, by: &[&str], descending: &[bool]) -> Self {
        LogicalPlan::Sort {
            input: Box::new(self),
            by: by.iter().map(|s| (*s).to_owned()).collect(),
            descending: descending.to_vec(),
        }
    }
    /// Keep the first `n` rows.
    pub fn limit(self, n: usize) -> Self {
        LogicalPlan::Limit {
            input: Box::new(self),
            n,
        }
    }

    /// Output column names, without executing (a CSV scan only reads its header).
    pub fn columns(&self) -> Result<Vec<String>, CoreError> {
        Ok(self.describe(false)?.into_iter().map(|(n, _)| n).collect())
    }

    /// Output schema. A CSV scan has no types until it is read, so this
    /// reads the file for plans that contain one; use [`LogicalPlan::columns`]
    /// to avoid that.
    pub fn schema(&self) -> Result<Schema, CoreError> {
        self.describe(true)?
            .into_iter()
            .map(|(n, d)| {
                d.map(|d| (n.clone(), d))
                    .ok_or_else(|| CoreError::Invalid(format!("type of column {n:?} is unknown")))
            })
            .collect()
    }

    /// Schema when every type is known without reading a file.
    pub(crate) fn known_schema(&self) -> Option<Schema> {
        self.describe(false)
            .ok()?
            .into_iter()
            .map(|(n, d)| d.map(|d| (n, d)))
            .collect()
    }

    /// Output names with their types (`None` = unknown until a CSV is read).
    fn describe(&self, read: bool) -> Result<Vec<(String, Option<DataType>)>, CoreError> {
        Ok(match self {
            LogicalPlan::Scan { df, projection, .. } => match projection {
                None => df
                    .columns()
                    .iter()
                    .map(|c| (c.name().to_owned(), Some(c.data_type())))
                    .collect(),
                Some(p) => p
                    .iter()
                    .map(|n| Ok((n.clone(), Some(df.column(n)?.data_type()))))
                    .collect::<Result<_, CoreError>>()?,
            },
            LogicalPlan::ScanCsv {
                path,
                options,
                header,
            } => {
                if read {
                    let df = read_csv(path, options).map_err(csv_err)?;
                    df.columns()
                        .iter()
                        .map(|c| (c.name().to_owned(), Some(c.data_type())))
                        .collect()
                } else {
                    options
                        .projection
                        .as_ref()
                        .unwrap_or(header)
                        .iter()
                        .map(|n| (n.clone(), None))
                        .collect()
                }
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => input.describe(read)?,
            LogicalPlan::Select { input, columns } => {
                let cols = input.describe(read)?;
                columns
                    .iter()
                    .map(|n| Ok((n.clone(), *find(&cols, n)?)))
                    .collect::<Result<_, CoreError>>()?
            }
            LogicalPlan::WithColumns { input, exprs } => {
                let mut cols = input.describe(read)?;
                for (name, e) in exprs {
                    let mut unknown = false;
                    for c in e.columns() {
                        unknown |= find(&cols, &c)?.is_none();
                    }
                    let dt = if unknown {
                        None
                    } else {
                        let look = |c: &str| {
                            find(&cols, c)?.ok_or_else(|| CoreError::ColumnNotFound(c.to_owned()))
                        };
                        Some(e.output_dtype_in(&look)?)
                    };
                    match cols.iter_mut().find(|(n, _)| n == name) {
                        Some(slot) => slot.1 = dt,
                        None => cols.push((name.clone(), dt)),
                    }
                }
                cols
            }
            LogicalPlan::GroupBy { input, keys, aggs } => {
                let cols = input.describe(read)?;
                let mut out = Vec::new();
                for k in keys {
                    out.push((k.clone(), *find(&cols, k)?));
                }
                for (c, kind) in aggs {
                    let dt = match *find(&cols, c)? {
                        None => None,
                        Some(d) => Some(match kind {
                            AggKind::Count => DataType::Int64,
                            AggKind::Mean if d.is_numeric() => DataType::Float64,
                            AggKind::Sum | AggKind::Min | AggKind::Max if d.is_numeric() => {
                                if d == DataType::Float64 {
                                    DataType::Float64
                                } else {
                                    DataType::Int64
                                }
                            }
                            _ => {
                                return Err(CoreError::ColumnNotFound(format!("{c} (not numeric)")))
                            }
                        }),
                    };
                    out.push((format!("{c}_{}", agg_suffix(*kind)), dt));
                }
                out
            }
            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                how,
            } => {
                let (l, r) = (left.describe(read)?, right.describe(read)?);
                if left_on.len() != right_on.len() || left_on.is_empty() {
                    return Err(CoreError::Invalid(
                        "join needs the same non-zero number of left and right keys".into(),
                    ));
                }
                let lnames: Vec<String> = l.iter().map(|(n, _)| n.clone()).collect();
                let mut out = l.clone();
                // Same-named keys: the right column is dropped; right/outer
                // joins fill the left key from it, which can change its type.
                let mut dropped: Vec<&String> = Vec::new();
                for (a, b) in left_on.iter().zip(right_on) {
                    let (ld, rd) = (*find(&l, a)?, *find(&r, b)?);
                    if a != b {
                        continue;
                    }
                    dropped.push(b);
                    if matches!(how, JoinType::Right | JoinType::Outer) {
                        let merged = match (ld, rd) {
                            (Some(x), Some(y)) => Some(merged_key_type(x, y).ok_or_else(|| {
                                CoreError::Invalid(format!(
                                    "right/outer join key {a:?} has incompatible types"
                                ))
                            })?),
                            _ => None,
                        };
                        if let Some(slot) = out.iter_mut().find(|(n, _)| n == a) {
                            slot.1 = merged;
                        }
                    }
                }
                for (n, d) in r.iter().filter(|(n, _)| !dropped.contains(&n)) {
                    out.push((
                        if lnames.contains(n) {
                            format!("{n}_right")
                        } else {
                            n.clone()
                        },
                        *d,
                    ));
                }
                out
            }
        })
    }

    /// Execute.
    pub fn collect(&self) -> Result<DataFrame, CoreError> {
        match self {
            LogicalPlan::Scan {
                df,
                projection,
                predicate,
            } => {
                let mask = match predicate {
                    Some(p) => Some(mask_of(p, df)?),
                    None => None,
                };
                let mut out = match projection {
                    Some(p) => {
                        let names: Vec<&str> = p.iter().map(String::as_str).collect();
                        df.select(&names)?
                    }
                    None => (**df).clone(),
                };
                if let Some(m) = mask {
                    out = out.filter(&m)?;
                }
                Ok(out)
            }
            LogicalPlan::ScanCsv { path, options, .. } => read_csv(path, options).map_err(csv_err),
            LogicalPlan::Filter { input, predicate } => {
                let df = input.collect()?;
                df.filter(&mask_of(predicate, &df)?)
            }
            LogicalPlan::Select { input, columns } => {
                let names: Vec<&str> = columns.iter().map(String::as_str).collect();
                input.collect()?.select(&names)
            }
            LogicalPlan::WithColumns { input, exprs } => {
                let mut df = input.collect()?;
                for (name, e) in exprs {
                    let arr = e.evaluate(&df)?;
                    df = df.with_column(Series::new(name.clone(), arr))?;
                }
                Ok(df)
            }
            LogicalPlan::GroupBy { input, keys, aggs } => {
                let df = input.collect()?;
                let k: Vec<&str> = keys.iter().map(String::as_str).collect();
                let a: Vec<(&str, AggKind)> =
                    aggs.iter().map(|(c, kind)| (c.as_str(), *kind)).collect();
                df.group_by(&k, &a)
            }
            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                how,
            } => {
                let (l, r) = (left.collect()?, right.collect()?);
                let lk: Vec<&str> = left_on.iter().map(String::as_str).collect();
                let rk: Vec<&str> = right_on.iter().map(String::as_str).collect();
                l.join_keys(&r, &lk, &rk, *how)
            }
            LogicalPlan::Sort {
                input,
                by,
                descending,
            } => {
                let b: Vec<&str> = by.iter().map(String::as_str).collect();
                input.collect()?.sort_by_multi(&b, descending)
            }
            LogicalPlan::Limit { input, n } => input.collect()?.head(*n),
        }
    }

    /// Plan before and after optimisation.
    pub fn explain(&self) -> String {
        format!(
            "== unoptimized ==\n{self}== optimized ==\n{}",
            self.clone().optimize()
        )
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        let pad = "  ".repeat(depth);
        match self {
            LogicalPlan::Scan {
                df,
                projection,
                predicate,
            } => {
                let cols = df.column_names().join(", ");
                write!(f, "{pad}SCAN [{cols}]")?;
                if let Some(p) = projection {
                    write!(f, " projection=[{}]", p.join(", "))?;
                }
                if let Some(p) = predicate {
                    write!(f, " filter={p}")?;
                }
                writeln!(f)
            }
            LogicalPlan::ScanCsv {
                path,
                options,
                header,
            } => {
                let shown = options.projection.as_ref().unwrap_or(header);
                writeln!(f, "{pad}SCAN CSV {} [{}]", path.display(), shown.join(", "))
            }
            LogicalPlan::Filter { input, predicate } => {
                writeln!(f, "{pad}FILTER {predicate}")?;
                input.fmt_indent(f, depth + 1)
            }
            LogicalPlan::Select { input, columns } => {
                writeln!(f, "{pad}SELECT [{}]", columns.join(", "))?;
                input.fmt_indent(f, depth + 1)
            }
            LogicalPlan::WithColumns { input, exprs } => {
                let list: Vec<String> = exprs.iter().map(|(n, e)| format!("{e} AS {n}")).collect();
                writeln!(f, "{pad}WITH_COLUMNS [{}]", list.join(", "))?;
                input.fmt_indent(f, depth + 1)
            }
            LogicalPlan::GroupBy { input, keys, aggs } => {
                let list: Vec<String> = aggs
                    .iter()
                    .map(|(c, k)| format!("{}({c})", agg_suffix(*k)))
                    .collect();
                writeln!(
                    f,
                    "{pad}GROUP_BY [{}] AGG [{}]",
                    keys.join(", "),
                    list.join(", ")
                )?;
                input.fmt_indent(f, depth + 1)
            }
            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                how,
            } => {
                let keys: Vec<String> = if left_on == right_on {
                    left_on.clone()
                } else {
                    left_on
                        .iter()
                        .zip(right_on)
                        .map(|(a, b)| format!("{a} = {b}"))
                        .collect()
                };
                writeln!(f, "{pad}JOIN {how:?} ON [{}]", keys.join(", "))?;
                left.fmt_indent(f, depth + 1)?;
                right.fmt_indent(f, depth + 1)
            }
            LogicalPlan::Sort {
                input,
                by,
                descending,
            } => {
                let list: Vec<String> = by
                    .iter()
                    .zip(descending)
                    .map(|(c, d)| format!("{c} {}", if *d { "DESC" } else { "ASC" }))
                    .collect();
                writeln!(f, "{pad}SORT [{}]", list.join(", "))?;
                input.fmt_indent(f, depth + 1)
            }
            LogicalPlan::Limit { input, n } => {
                writeln!(f, "{pad}LIMIT {n}")?;
                input.fmt_indent(f, depth + 1)
            }
        }
    }
}

fn mask_of(predicate: &Expr, df: &DataFrame) -> Result<tessera_arrow::BooleanArray, CoreError> {
    match predicate.evaluate(df)? {
        Array::Boolean(mask) => Ok(mask),
        _ => Err(CoreError::ColumnNotFound(format!(
            "predicate {predicate} is not boolean"
        ))),
    }
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}
