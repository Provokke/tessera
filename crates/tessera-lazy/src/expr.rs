//! Expressions and their evaluation: a vectorized evaluator built on the
//! `tessera-compute` kernels, and a row-wise reference evaluator.
use std::collections::BTreeSet;
use std::fmt;

use tessera_arrow::{Array, BooleanArray, PrimitiveArray};
use tessera_compute::arithmetic::{
    add_f64, add_i64, binary, div_f64, div_i64, mul_i64, sub_f64, sub_i64,
};
use tessera_compute::boolean;
use tessera_compute::comparison::{cmp, cmp_scalar, cmp_utf8, cmp_utf8_scalar, CmpOp};
use tessera_core::{CoreError, DataFrame, DataType, Scalar};

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (integer division by zero gives null; floats follow IEEE-754)
    Div,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// logical and (null if either side null)
    And,
    /// logical or (null if either side null)
    Or,
}

impl Op {
    /// Whether the operator always produces a boolean.
    pub fn is_boolean(self) -> bool {
        !matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)
    }
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference.
    Col(String),
    /// Literal.
    Lit(Scalar),
    /// Binary operation.
    Binary(Box<Expr>, Op, Box<Expr>),
    /// Logical negation (null stays null); the operand must be boolean.
    Not(Box<Expr>),
    /// Name the result of an expression (used by `with_columns`).
    Alias(Box<Expr>, String),
}

/// Column reference.
pub fn col(name: &str) -> Expr {
    Expr::Col(name.to_owned())
}
/// Literal i64.
pub fn lit_i64(v: i64) -> Expr {
    Expr::Lit(Scalar::Int64(v))
}
/// Literal f64.
pub fn lit_f64(v: f64) -> Expr {
    Expr::Lit(Scalar::Float64(v))
}
/// Literal string.
pub fn lit_str(v: &str) -> Expr {
    Expr::Lit(Scalar::Utf8(v.to_owned()))
}
/// Literal boolean.
pub fn lit_bool(v: bool) -> Expr {
    Expr::Lit(Scalar::Boolean(v))
}

/// A vectorized intermediate: a column or a scalar to broadcast.
enum Val {
    Arr(Array),
    Lit(Scalar),
}

type Lookup<'a> = &'a dyn Fn(&str) -> Result<DataType, CoreError>;

impl Expr {
    fn bin(self, op: Op, rhs: Expr) -> Expr {
        Expr::Binary(Box::new(self), op, Box::new(rhs))
    }
    /// `self == rhs`
    pub fn equals(self, rhs: Expr) -> Expr {
        self.bin(Op::Eq, rhs)
    }
    /// `self < rhs`
    pub fn lt(self, rhs: Expr) -> Expr {
        self.bin(Op::Lt, rhs)
    }
    /// `self > rhs`
    pub fn gt(self, rhs: Expr) -> Expr {
        self.bin(Op::Gt, rhs)
    }
    /// `self != rhs`
    pub fn not_equals(self, rhs: Expr) -> Expr {
        self.bin(Op::Ne, rhs)
    }
    /// `self <= rhs`
    pub fn le(self, rhs: Expr) -> Expr {
        self.bin(Op::Le, rhs)
    }
    /// `self >= rhs`
    pub fn ge(self, rhs: Expr) -> Expr {
        self.bin(Op::Ge, rhs)
    }
    /// `NOT self`
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }
    /// `self AND rhs`
    pub fn and(self, rhs: Expr) -> Expr {
        self.bin(Op::And, rhs)
    }
    /// `self OR rhs`
    pub fn or(self, rhs: Expr) -> Expr {
        self.bin(Op::Or, rhs)
    }
    /// Name the result.
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), name.to_owned())
    }
    /// Build a binary node from an explicit operator.
    pub fn binary(self, op: Op, rhs: Expr) -> Expr {
        self.bin(op, rhs)
    }

    /// The expression without any outer alias.
    pub fn unaliased(&self) -> &Expr {
        match self {
            Expr::Alias(e, _) => e.unaliased(),
            e => e,
        }
    }

    /// Output name: the alias, or the column name of a bare column reference.
    pub fn output_name(&self) -> Option<&str> {
        match self {
            Expr::Alias(_, n) => Some(n),
            Expr::Col(c) => Some(c),
            _ => None,
        }
    }

    /// Whether the expression is statically known to produce booleans.
    pub fn is_boolean(&self) -> bool {
        match self {
            Expr::Lit(Scalar::Boolean(_)) => true,
            Expr::Binary(_, op, _) => op.is_boolean(),
            Expr::Not(_) => true,
            Expr::Alias(e, _) => e.is_boolean(),
            _ => false,
        }
    }

    /// Whether evaluation can fail depending on the data: `NOT` over an
    /// operand that is not statically boolean raises on the first non-null
    /// non-boolean value. Rewrites that change which rows an expression sees
    /// (pushdown below a join, group-by, ...) must not move such expressions.
    pub(crate) fn may_fail(&self) -> bool {
        match self {
            Expr::Col(_) | Expr::Lit(_) => false,
            Expr::Alias(e, _) => e.may_fail(),
            Expr::Binary(l, _, r) => l.may_fail() || r.may_fail(),
            Expr::Not(e) => !e.is_boolean() || e.may_fail(),
        }
    }

    /// Whether the expression contains a division.
    pub(crate) fn has_division(&self) -> bool {
        match self {
            Expr::Col(_) | Expr::Lit(_) => false,
            Expr::Alias(e, _) | Expr::Not(e) => e.has_division(),
            Expr::Binary(l, op, r) => *op == Op::Div || l.has_division() || r.has_division(),
        }
    }

    /// Names of all referenced columns.
    pub fn columns(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Col(c) => {
                out.insert(c.clone());
            }
            Expr::Lit(_) => {}
            Expr::Binary(l, _, r) => {
                l.collect_columns(out);
                r.collect_columns(out);
            }
            Expr::Not(e) | Expr::Alias(e, _) => e.collect_columns(out),
        }
    }

    /// Result type given a column-type lookup (`Int32` columns are reported as
    /// `Int32`; see [`Expr::output_dtype_in`] for the type of the evaluated array).
    pub(crate) fn dtype_in(&self, look: Lookup<'_>) -> Result<DataType, CoreError> {
        Ok(match self {
            Expr::Col(c) => look(c)?,
            Expr::Lit(s) => s.data_type(),
            Expr::Alias(e, _) => e.dtype_in(look)?,
            Expr::Not(_) => DataType::Boolean,
            Expr::Binary(l, op, r) => match op {
                Op::Add | Op::Sub | Op::Mul | Op::Div => {
                    if l.dtype_in(look)? == DataType::Float64
                        || r.dtype_in(look)? == DataType::Float64
                    {
                        DataType::Float64
                    } else {
                        DataType::Int64
                    }
                }
                _ => DataType::Boolean,
            },
        })
    }

    /// Type of the array [`Expr::evaluate`] returns (`Int32` widens to `Int64`).
    pub(crate) fn output_dtype_in(&self, look: Lookup<'_>) -> Result<DataType, CoreError> {
        Ok(match self.dtype_in(look)? {
            DataType::Int32 => DataType::Int64,
            d => d,
        })
    }

    fn dtype(&self, df: &DataFrame) -> Result<DataType, CoreError> {
        self.dtype_in(&|c| Ok(df.column(c)?.data_type()))
    }

    /// Constant-fold a binary node whose operands are both non-null literals.
    /// Returns `None` unless the result has the expression's static type.
    pub(crate) fn fold_constant(&self) -> Option<Scalar> {
        match self {
            Expr::Binary(l, _, r) => {
                let (Expr::Lit(a), Expr::Lit(b)) = (l.as_ref(), r.as_ref()) else {
                    return None;
                };
                if a.is_null() || b.is_null() {
                    return None;
                }
            }
            Expr::Not(e) => {
                if !matches!(e.as_ref(), Expr::Lit(Scalar::Boolean(_))) {
                    return None;
                }
            }
            _ => return None,
        }
        let empty = DataFrame::new(vec![]).ok()?;
        let want = self.dtype(&empty).ok()?;
        let s = self.eval_row(&empty, 0).ok()?;
        (!s.is_null() && s.data_type() == want).then_some(s)
    }

    fn eval_row(&self, df: &DataFrame, i: usize) -> Result<Scalar, CoreError> {
        Ok(match self {
            Expr::Col(c) => df.column(c)?.get(i),
            Expr::Lit(s) => s.clone(),
            Expr::Alias(e, _) => e.eval_row(df, i)?,
            Expr::Not(e) => match e.eval_row(df, i)? {
                Scalar::Boolean(b) => Scalar::Boolean(!b),
                s if s.is_null() => Scalar::Null(DataType::Boolean),
                _ => return Err(CoreError::Invalid("NOT needs a boolean operand".into())),
            },
            Expr::Binary(l, op, r) => {
                let (a, b) = (l.eval_row(df, i)?, r.eval_row(df, i)?);
                let dt = self.dtype(df)?;
                if a.is_null() || b.is_null() {
                    return Ok(Scalar::Null(dt));
                }
                let int = |s: &Scalar| match s {
                    Scalar::Int32(v) => Some(*v as i64),
                    Scalar::Int64(v) => Some(*v),
                    _ => None,
                };
                let boolean = |s: &Scalar| matches!(s, Scalar::Boolean(true));
                match op {
                    Op::And => Scalar::Boolean(boolean(&a) && boolean(&b)),
                    Op::Or => Scalar::Boolean(boolean(&a) || boolean(&b)),
                    Op::Add | Op::Sub | Op::Mul | Op::Div => match (int(&a), int(&b), dt) {
                        (Some(_), Some(0), DataType::Int64) if *op == Op::Div => {
                            Scalar::Null(DataType::Int64)
                        }
                        (Some(x), Some(y), DataType::Int64) => Scalar::Int64(match op {
                            Op::Add => x.wrapping_add(y),
                            Op::Sub => x.wrapping_sub(y),
                            Op::Div => x.wrapping_div(y),
                            _ => x.wrapping_mul(y),
                        }),
                        _ => {
                            let (x, y) = (
                                a.as_f64().unwrap_or(f64::NAN),
                                b.as_f64().unwrap_or(f64::NAN),
                            );
                            Scalar::Float64(match op {
                                Op::Add => x + y,
                                Op::Sub => x - y,
                                Op::Div => x / y,
                                _ => x * y,
                            })
                        }
                    },
                    Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                        let ord = match (int(&a), int(&b)) {
                            (Some(x), Some(y)) => Some(x.cmp(&y)),
                            _ => match (&a, &b) {
                                (Scalar::Utf8(x), Scalar::Utf8(y)) => Some(x.cmp(y)),
                                (Scalar::Boolean(x), Scalar::Boolean(y)) => Some(x.cmp(y)),
                                _ => a
                                    .as_f64()
                                    .zip(b.as_f64())
                                    .and_then(|(x, y)| x.partial_cmp(&y)),
                            },
                        };
                        use std::cmp::Ordering::{Equal, Greater, Less};
                        Scalar::Boolean(match op {
                            Op::Eq => ord == Some(Equal),
                            Op::Ne => ord != Some(Equal),
                            Op::Lt => ord == Some(Less),
                            Op::Le => matches!(ord, Some(Less | Equal)),
                            Op::Gt => ord == Some(Greater),
                            _ => matches!(ord, Some(Greater | Equal)),
                        })
                    }
                }
            }
        })
    }

    /// Evaluate a sub-expression with kernels. `None` means "not supported
    /// here": the caller falls back to row-wise evaluation.
    fn vec_val(&self, df: &DataFrame) -> Result<Option<Val>, CoreError> {
        Ok(match self {
            Expr::Col(c) => Some(Val::Arr(df.column(c)?.array().clone())),
            Expr::Lit(s) if !s.is_null() => Some(Val::Lit(s.clone())),
            Expr::Lit(_) => None,
            Expr::Alias(e, _) => e.vec_val(df)?,
            Expr::Not(e) => match e.vec_val(df)? {
                Some(Val::Arr(Array::Boolean(x))) => {
                    Some(Val::Arr(Array::Boolean(boolean::not(&x)?)))
                }
                _ => None,
            },
            Expr::Binary(l, op, r) => {
                let (Some(a), Some(b)) = (l.vec_val(df)?, r.vec_val(df)?) else {
                    return Ok(None);
                };
                let out = match op {
                    Op::And | Op::Or => match (a, b) {
                        (Val::Arr(Array::Boolean(x)), Val::Arr(Array::Boolean(y))) => {
                            Some(Array::Boolean(if *op == Op::And {
                                boolean::and(&x, &y)?
                            } else {
                                boolean::or(&x, &y)?
                            }))
                        }
                        _ => None,
                    },
                    Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                        let cop = match op {
                            Op::Eq => CmpOp::Eq,
                            Op::Ne => CmpOp::NotEq,
                            Op::Lt => CmpOp::Lt,
                            Op::Le => CmpOp::LtEq,
                            Op::Gt => CmpOp::Gt,
                            _ => CmpOp::GtEq,
                        };
                        vec_cmp(a, cop, b)?.map(Array::Boolean)
                    }
                    Op::Add | Op::Sub | Op::Mul | Op::Div => vec_arith(a, *op, b, df.height())?,
                };
                out.map(Val::Arr)
            }
        })
    }

    /// Evaluate with the vectorized kernels only. Returns `Ok(None)` when the
    /// expression (or one of its operand types) is not covered, in which case
    /// [`Expr::evaluate`] uses row-wise evaluation.
    ///
    /// Covered: `==`/`<`/`>` between Int32/Int64/Float64 columns and literals
    /// (mixed int/float compares as f64) and between Utf8 columns/literals;
    /// `+`/`-`/`*`/`/` over Int32/Int64/Float64 (integer `/ 0` is null);
    /// `NOT` over boolean operands; `AND`/`OR` over boolean operands.
    pub fn evaluate_vectorized(&self, df: &DataFrame) -> Result<Option<Array>, CoreError> {
        if !matches!(self.unaliased(), Expr::Binary(..) | Expr::Not(_)) {
            return Ok(None);
        }
        Ok(match self.vec_val(df)? {
            Some(Val::Arr(a)) => Some(a),
            _ => None,
        })
    }

    /// Evaluate against every row of `df`.
    pub fn evaluate(&self, df: &DataFrame) -> Result<Array, CoreError> {
        if let Some(a) = self.evaluate_vectorized(df)? {
            return Ok(a);
        }
        self.evaluate_rowwise(df)
    }

    /// Reference evaluation, always row by row (used to test the fast path).
    pub fn evaluate_rowwise(&self, df: &DataFrame) -> Result<Array, CoreError> {
        let rows = (0..df.height())
            .map(|i| self.eval_row(df, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match self.dtype(df)? {
            DataType::Boolean => {
                Array::Boolean(BooleanArray::from_options(rows.iter().map(|s| match s {
                    Scalar::Boolean(b) => Some(*b),
                    _ => None,
                })))
            }
            DataType::Float64 => Array::Float64(PrimitiveArray::from_options(
                rows.iter().map(Scalar::as_f64),
            )),
            DataType::Int32 | DataType::Int64 => {
                Array::Int64(PrimitiveArray::from_options(rows.iter().map(|s| match s {
                    Scalar::Int32(v) => Some(*v as i64),
                    Scalar::Int64(v) => Some(*v),
                    _ => None,
                })))
            }
            DataType::Utf8 => match self.unaliased() {
                Expr::Col(c) => df.column(c)?.array().clone(),
                _ => return Err(CoreError::ColumnNotFound("string expression".into())),
            },
        })
    }
}

fn to_i64(a: &Array) -> Result<Option<PrimitiveArray<i64>>, CoreError> {
    Ok(match a {
        Array::Int64(x) => Some(x.clone()),
        Array::Int32(x) => Some(PrimitiveArray::try_new(
            x.values().iter().map(|&v| v as i64).collect(),
            x.validity().cloned(),
        )?),
        _ => None,
    })
}

fn to_f64(a: &Array) -> Result<Option<PrimitiveArray<f64>>, CoreError> {
    Ok(match a {
        Array::Float64(x) => Some(x.clone()),
        Array::Int64(x) => Some(PrimitiveArray::try_new(
            x.values().iter().map(|&v| v as f64).collect(),
            x.validity().cloned(),
        )?),
        Array::Int32(x) => Some(PrimitiveArray::try_new(
            x.values().iter().map(|&v| v as f64).collect(),
            x.validity().cloned(),
        )?),
        _ => None,
    })
}

fn is_int(a: &Array) -> bool {
    matches!(a, Array::Int32(_) | Array::Int64(_))
}

fn int_lit(s: &Scalar) -> Option<i64> {
    match s {
        Scalar::Int32(v) => Some(*v as i64),
        Scalar::Int64(v) => Some(*v),
        _ => None,
    }
}

fn flip(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::LtEq => CmpOp::GtEq,
        CmpOp::GtEq => CmpOp::LtEq,
        o => o,
    }
}

fn vec_cmp(a: Val, op: CmpOp, b: Val) -> Result<Option<BooleanArray>, CoreError> {
    match (a, b) {
        (Val::Lit(_), Val::Lit(_)) => Ok(None),
        (Val::Lit(k), arr @ Val::Arr(_)) => vec_cmp(arr, flip(op), Val::Lit(k)),
        (Val::Arr(x), Val::Lit(k)) => Ok(match (&x, &k) {
            (Array::Utf8(s), Scalar::Utf8(t)) => Some(cmp_utf8_scalar(s, op, t)?),
            _ if is_int(&x) && int_lit(&k).is_some() => match (to_i64(&x)?, int_lit(&k)) {
                (Some(v), Some(k)) => Some(cmp_scalar(&v, op, k)),
                _ => None,
            },
            (Array::Float64(_) | Array::Int32(_) | Array::Int64(_), _) => {
                match (to_f64(&x)?, k.as_f64()) {
                    (Some(v), Some(k)) => Some(cmp_scalar(&v, op, k)),
                    _ => None,
                }
            }
            _ => None,
        }),
        (Val::Arr(x), Val::Arr(y)) => Ok(match (&x, &y) {
            (Array::Utf8(s), Array::Utf8(t)) => Some(cmp_utf8(s, op, t)?),
            _ if is_int(&x) && is_int(&y) => match (to_i64(&x)?, to_i64(&y)?) {
                (Some(p), Some(q)) => Some(cmp(&p, op, &q)?),
                _ => None,
            },
            _ => match (to_f64(&x)?, to_f64(&y)?) {
                (Some(p), Some(q)) => Some(cmp(&p, op, &q)?),
                _ => None,
            },
        }),
    }
}

/// Broadcast a literal (or pass a column through) as an i64 array.
fn int_operand(v: &Val, n: usize) -> Result<Option<PrimitiveArray<i64>>, CoreError> {
    Ok(match v {
        Val::Arr(a) => to_i64(a)?,
        Val::Lit(s) => match int_lit(s) {
            Some(k) => Some(PrimitiveArray::try_new((0..n).map(|_| k).collect(), None)?),
            None => None,
        },
    })
}

/// Broadcast a literal (or pass a column through) as an f64 array.
fn float_operand(v: &Val, n: usize) -> Result<Option<PrimitiveArray<f64>>, CoreError> {
    Ok(match v {
        Val::Arr(a) => to_f64(a)?,
        Val::Lit(s) => match s.as_f64() {
            Some(k) => Some(PrimitiveArray::try_new((0..n).map(|_| k).collect(), None)?),
            None => None,
        },
    })
}

fn vec_arith(a: Val, op: Op, b: Val, n: usize) -> Result<Option<Array>, CoreError> {
    if matches!((&a, &b), (Val::Lit(_), Val::Lit(_))) {
        return Ok(None);
    }
    let is_float = |v: &Val| match v {
        Val::Arr(x) => matches!(x, Array::Float64(_)),
        Val::Lit(s) => matches!(s, Scalar::Float64(_)),
    };
    if is_float(&a) || is_float(&b) {
        let (Some(x), Some(y)) = (float_operand(&a, n)?, float_operand(&b, n)?) else {
            return Ok(None);
        };
        Ok(Some(Array::Float64(match op {
            Op::Add => add_f64(&x, &y)?,
            Op::Sub => sub_f64(&x, &y)?,
            Op::Div => div_f64(&x, &y)?,
            _ => binary(&x, &y, |p, q| p * q)?,
        })))
    } else {
        let (Some(x), Some(y)) = (int_operand(&a, n)?, int_operand(&b, n)?) else {
            return Ok(None);
        };
        Ok(Some(Array::Int64(match op {
            Op::Add => add_i64(&x, &y)?,
            Op::Sub => sub_i64(&x, &y)?,
            Op::Div => div_i64(&x, &y)?,
            _ => mul_i64(&x, &y)?,
        })))
    }
}

macro_rules! arith {
    ($tr:ident, $m:ident, $op:ident) => {
        impl std::ops::$tr for Expr {
            type Output = Expr;
            fn $m(self, rhs: Expr) -> Expr {
                self.bin(Op::$op, rhs)
            }
        }
    };
}
arith!(Add, add, Add);
arith!(Sub, sub, Sub);
arith!(Mul, mul, Mul);
arith!(Div, div, Div);

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Col(c) => write!(f, "col({c})"),
            Expr::Lit(s) => write!(f, "{s}"),
            Expr::Alias(e, n) => write!(f, "{e} AS {n}"),
            Expr::Not(e) => write!(f, "NOT {e}"),
            Expr::Binary(l, op, r) => {
                let o = match op {
                    Op::Add => "+",
                    Op::Sub => "-",
                    Op::Mul => "*",
                    Op::Div => "/",
                    Op::Eq => "==",
                    Op::Ne => "!=",
                    Op::Lt => "<",
                    Op::Le => "<=",
                    Op::Gt => ">",
                    Op::Ge => ">=",
                    Op::And => "AND",
                    Op::Or => "OR",
                };
                write!(f, "({l} {o} {r})")
            }
        }
    }
}
