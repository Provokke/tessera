//! Python bindings: eager `DataFrame` (dict/CSV constructors, select, filter,
//! with_columns, join, sort, head, group_by().agg, to_dict, zero-copy NumPy
//! export through the buffer protocol), `Expr` objects with operator
//! overloading, and a `LazyFrame` (`df.lazy()` / `scan_csv`, `.collect()`).
// The only hand-written unsafe is the buffer-protocol implementation of
// `ColumnView`; `unsafe_op_in_unsafe_fn` is not denied because pyo3 0.22 macro
// expansions trip it.
#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(missing_docs)]
// pyo3 0.22 #[pymethods] expansion triggers this lint on PyResult returns.
#![allow(clippy::useless_conversion)]

use std::ffi::{c_char, c_int, c_void};

use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use tessera_arrow::{Array, BooleanArray, Buffer, PrimitiveArray, Utf8Array};
use tessera_core::{AggKind, DataFrame, JoinType, Scalar, Series};
use tessera_io::CsvOptions;
use tessera_lazy::{col, lit_bool, lit_f64, lit_i64, lit_str, Expr, LogicalPlan};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn series_from_list(name: &str, list: &Bound<'_, PyList>) -> PyResult<Series> {
    let items: Vec<Bound<'_, PyAny>> = list.iter().collect();
    let first = items.iter().find(|v| !v.is_none());
    let array = match first {
        Some(v) if v.is_instance_of::<PyBool>() => Array::Boolean(BooleanArray::from_options(
            items
                .iter()
                .map(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract::<bool>().map(Some)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?,
        )),
        Some(v) if v.is_instance_of::<PyInt>() => Array::Int64(PrimitiveArray::from_options(
            items
                .iter()
                .map(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract::<i64>().map(Some)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?,
        )),
        Some(v) if v.is_instance_of::<PyFloat>() => Array::Float64(PrimitiveArray::from_options(
            items
                .iter()
                .map(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract::<f64>().map(Some)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?,
        )),
        _ => {
            let strs = items
                .iter()
                .map(|v| {
                    if v.is_none() {
                        Ok(None)
                    } else {
                        v.extract::<String>().map(Some)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            Array::Utf8(Utf8Array::from_options(strs.iter().map(|s| s.as_deref())).map_err(err)?)
        }
    };
    Ok(Series::new(name, array))
}

/// A string or a list of strings.
fn str_or_list(v: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(vec![s]);
    }
    v.extract::<Vec<String>>()
        .map_err(|_| PyTypeError::new_err("expected a column name or a list of column names"))
}

fn bool_or_list(v: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<bool>> {
    if let Ok(b) = v.extract::<bool>() {
        return Ok(vec![b; n]);
    }
    let list = v
        .extract::<Vec<bool>>()
        .map_err(|_| PyTypeError::new_err("descending must be a bool or a list of bools"))?;
    if list.len() != n {
        return Err(PyValueError::new_err(
            "descending needs one flag per sort column",
        ));
    }
    Ok(list)
}

fn parse_how(how: &str) -> PyResult<JoinType> {
    match how {
        "inner" => Ok(JoinType::Inner),
        "left" => Ok(JoinType::Left),
        "right" => Ok(JoinType::Right),
        "outer" => Ok(JoinType::Outer),
        other => Err(PyValueError::new_err(format!(
            "unsupported join type {other:?} (use \"inner\", \"left\", \"right\" or \"outer\")"
        ))),
    }
}

/// Join keys from `on` (same names on both sides) or from `left_on` and
/// `right_on` (paired by position).
fn join_key_args(
    on: Option<&Bound<'_, PyAny>>,
    left_on: Option<&Bound<'_, PyAny>>,
    right_on: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<String>, Vec<String>)> {
    match (on, left_on, right_on) {
        (Some(o), None, None) => {
            let k = str_or_list(o)?;
            Ok((k.clone(), k))
        }
        (None, Some(l), Some(r)) => {
            let (l, r) = (str_or_list(l)?, str_or_list(r)?);
            if l.len() != r.len() {
                return Err(PyValueError::new_err(
                    "left_on and right_on must have the same number of columns",
                ));
            }
            Ok((l, r))
        }
        _ => Err(PyValueError::new_err(
            "pass either on, or both left_on and right_on",
        )),
    }
}

fn parse_aggs(aggs: Vec<(String, String)>) -> PyResult<Vec<(String, AggKind)>> {
    aggs.into_iter()
        .map(|(c, k)| {
            let kind = match k.as_str() {
                "sum" => AggKind::Sum,
                "mean" => AggKind::Mean,
                "min" => AggKind::Min,
                "max" => AggKind::Max,
                "count" => AggKind::Count,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown aggregation {other}"
                    )))
                }
            };
            Ok((c, kind))
        })
        .collect()
}

fn csv_options(infer_rows: Option<usize>, columns: Option<Vec<String>>) -> CsvOptions {
    let mut o = CsvOptions::new();
    if let Some(n) = infer_rows {
        o = o.with_infer_rows(n);
    }
    if let Some(c) = columns {
        o = o.with_projection(c);
    }
    o
}

/// Python value to an expression: an `Expr`, or a bool/int/float/str literal.
fn to_expr(v: &Bound<'_, PyAny>) -> PyResult<Expr> {
    if let Ok(e) = v.extract::<PyRef<'_, PyExpr>>() {
        return Ok(e.e.clone());
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(lit_bool(v.extract()?));
    }
    if v.is_instance_of::<PyInt>() {
        return Ok(lit_i64(v.extract()?));
    }
    if v.is_instance_of::<PyFloat>() {
        return Ok(lit_f64(v.extract()?));
    }
    if v.is_instance_of::<PyString>() {
        return Ok(lit_str(&v.extract::<String>()?));
    }
    Err(PyTypeError::new_err(
        "expected an Expr or a bool/int/float/str literal",
    ))
}

fn exprs_from(list: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Expr>> {
    list.iter().map(to_expr).collect()
}

/// A column expression. Build with `col("a")` / `lit(1)` and Python operators
/// (`+ - *`, `> >= < <= == !=`, `&` for AND, `|` for OR, `~` for NOT); name with `.alias("x")`.
#[pyclass(name = "Expr")]
#[derive(Clone)]
struct PyExpr {
    e: Expr,
}

impl PyExpr {
    fn bin(&self, other: &Bound<'_, PyAny>, f: impl FnOnce(Expr, Expr) -> Expr) -> PyResult<Self> {
        Ok(PyExpr {
            e: f(self.e.clone(), to_expr(other)?),
        })
    }
    fn rbin(&self, other: &Bound<'_, PyAny>, f: impl FnOnce(Expr, Expr) -> Expr) -> PyResult<Self> {
        Ok(PyExpr {
            e: f(to_expr(other)?, self.e.clone()),
        })
    }
}

#[pymethods]
impl PyExpr {
    fn alias(&self, name: &str) -> PyExpr {
        PyExpr {
            e: self.e.clone().alias(name),
        }
    }
    fn __add__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.bin(o, |a, b| a + b)
    }
    fn __radd__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.rbin(o, |a, b| a + b)
    }
    fn __sub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.bin(o, |a, b| a - b)
    }
    fn __rsub__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.rbin(o, |a, b| a - b)
    }
    fn __mul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.bin(o, |a, b| a * b)
    }
    fn __rmul__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.rbin(o, |a, b| a * b)
    }
    fn __and__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.bin(o, Expr::and)
    }
    fn __rand__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.rbin(o, Expr::and)
    }
    fn __or__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.bin(o, Expr::or)
    }
    fn __ror__(&self, o: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        self.rbin(o, Expr::or)
    }
    /// `NOT` (the `~` operator); the operand must be boolean.
    fn __invert__(&self) -> PyExpr {
        PyExpr {
            e: self.e.clone().not(),
        }
    }
    /// All six comparison operators build expressions.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        o: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<PyObject> {
        let e = match op {
            CompareOp::Gt => self.bin(o, Expr::gt)?,
            CompareOp::Lt => self.bin(o, Expr::lt)?,
            CompareOp::Eq => self.bin(o, Expr::equals)?,
            CompareOp::Ne => self.bin(o, Expr::not_equals)?,
            CompareOp::Le => self.bin(o, Expr::le)?,
            CompareOp::Ge => self.bin(o, Expr::ge)?,
        };
        Ok(e.into_py(py))
    }
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "an Expr has no truth value; combine expressions with & and |",
        ))
    }
    fn __repr__(&self) -> String {
        self.e.to_string()
    }
}

/// Column reference expression.
#[pyfunction(name = "col")]
fn py_col(name: &str) -> PyExpr {
    PyExpr { e: col(name) }
}

/// Literal expression (bool, int, float or str).
#[pyfunction]
fn lit(value: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(PyExpr { e: to_expr(value)? })
}

/// The values of a numeric column exposed through the Python buffer protocol.
/// Holds a clone of the column's `Buffer<T>` (an `Arc` plus offset), so
/// `numpy.asarray(view)` shares the memory with the DataFrame: no copy. The
/// buffer is read-only. Slots that are null hold unspecified values; use
/// `null_count` to know whether any exist.
#[pyclass(name = "ColumnView")]
struct PyColumnView {
    data: ViewData,
    /// Element count, as the `Py_ssize_t` the exported `shape` points to.
    shape: ffi::Py_ssize_t,
    nulls: usize,
}

enum ViewData {
    I64(Buffer<i64>),
    I32(Buffer<i32>),
    F64(Buffer<f64>),
}

impl PyColumnView {
    /// (pointer, itemsize, struct-module format).
    fn raw(&self) -> (*const c_void, usize, &'static std::ffi::CStr) {
        match &self.data {
            ViewData::I64(b) => (b.as_ptr().cast(), 8, c"q"),
            ViewData::I32(b) => (b.as_ptr().cast(), 4, c"i"),
            ViewData::F64(b) => (b.as_ptr().cast(), 8, c"d"),
        }
    }
}

#[pymethods]
impl PyColumnView {
    /// Address of the first element in the Rust buffer.
    fn data_ptr(&self) -> usize {
        self.raw().0 as usize
    }
    #[getter]
    fn null_count(&self) -> usize {
        self.nulls
    }
    fn __len__(&self) -> usize {
        self.shape as usize
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("the column is read-only"));
        }
        let this = slf.borrow();
        let (ptr, itemsize, format) = this.raw();
        // SAFETY: `view` is non-null and points to a `Py_buffer` that CPython
        // owns for the duration of this call. `buf` points into the `Buffer`
        // held by `this`, which stays alive (and immutable) for as long as the
        // `obj` reference stored below keeps the Python object alive.
        // `shape`/`strides` point at fields of that same Python object, and
        // `format` at a static C string.
        unsafe {
            (*view).obj = slf.clone().into_ptr();
            (*view).buf = ptr as *mut c_void;
            (*view).len = this.shape * itemsize as ffi::Py_ssize_t;
            (*view).readonly = 1;
            (*view).itemsize = itemsize as ffi::Py_ssize_t;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                format.as_ptr() as *mut c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                std::ptr::addr_of!(this.shape) as *mut ffi::Py_ssize_t
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                std::ptr::addr_of_mut!((*view).itemsize)
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Nothing owned by the view: `obj` is released by CPython.
    }
}

/// A tessera DataFrame.
#[pyclass(name = "DataFrame")]
struct PyDataFrame {
    df: DataFrame,
}

impl PyDataFrame {
    fn run(py: Python<'_>, plan: LogicalPlan) -> PyResult<PyDataFrame> {
        let df = py
            .allow_threads(move || plan.optimize().collect())
            .map_err(err)?;
        Ok(PyDataFrame { df })
    }
}

#[pymethods]
impl PyDataFrame {
    #[new]
    fn new(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cols = Vec::new();
        for (k, v) in data.iter() {
            let name: String = k.extract()?;
            let list = v.downcast::<PyList>()?;
            cols.push(series_from_list(&name, list)?);
        }
        Ok(PyDataFrame {
            df: DataFrame::new(cols).map_err(err)?,
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.df
            .column_names()
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.df.height()
    }

    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let names: Vec<&str> = columns.iter().map(String::as_str).collect();
        Ok(PyDataFrame {
            df: self.df.select(&names).map_err(err)?,
        })
    }

    /// `filter(expr)` with an `Expr`, or the legacy `filter(column, op, int)`
    /// with op in `>`, `<`, `==`.
    #[pyo3(signature = (predicate, op = None, value = None))]
    fn filter(
        &self,
        py: Python<'_>,
        predicate: &Bound<'_, PyAny>,
        op: Option<&str>,
        value: Option<i64>,
    ) -> PyResult<Self> {
        let pred = match (op, value) {
            (None, None) => to_expr(predicate)?,
            (Some(op), Some(value)) => {
                let column: String = predicate.extract()?;
                let rhs = lit_i64(value);
                match op {
                    ">" => col(&column).gt(rhs),
                    "<" => col(&column).lt(rhs),
                    "==" => col(&column).equals(rhs),
                    _ => return Err(PyValueError::new_err(format!("unsupported op {op}"))),
                }
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "filter takes an Expr, or (column, op, value)",
                ))
            }
        };
        Self::run(py, LogicalPlan::scan(self.df.clone()).filter(pred))
    }

    /// Add or replace columns from expressions (name them with `.alias`).
    fn with_columns(&self, py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let plan = LogicalPlan::scan(self.df.clone()).with_columns(exprs_from(exprs)?);
        Self::run(py, plan)
    }

    /// Hash join; `how` is "inner", "left", "right" or "outer". Keys are `on`
    /// (same names on both sides, the right key column is dropped) or
    /// `left_on` and `right_on` (both key columns are kept).
    #[pyo3(signature = (other, on = None, how = "inner", left_on = None, right_on = None))]
    fn join(
        &self,
        py: Python<'_>,
        other: &PyDataFrame,
        on: Option<&Bound<'_, PyAny>>,
        how: &str,
        left_on: Option<&Bound<'_, PyAny>>,
        right_on: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (lk, rk) = join_key_args(on, left_on, right_on)?;
        let how = parse_how(how)?;
        let (l, r) = (self.df.clone(), other.df.clone());
        let df = py
            .allow_threads(move || {
                let lk: Vec<&str> = lk.iter().map(String::as_str).collect();
                let rk: Vec<&str> = rk.iter().map(String::as_str).collect();
                l.join_keys(&r, &lk, &rk, how)
            })
            .map_err(err)?;
        Ok(PyDataFrame { df })
    }

    /// First `n` rows (zero-copy).
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> PyResult<Self> {
        Ok(PyDataFrame {
            df: self.df.head(n).map_err(err)?,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for c in self.df.columns() {
            let vals: Vec<PyObject> = (0..c.len())
                .map(|i| match c.get(i) {
                    Scalar::Null(_) => py.None(),
                    Scalar::Boolean(v) => v.into_py(py),
                    Scalar::Int32(v) => v.into_py(py),
                    Scalar::Int64(v) => v.into_py(py),
                    Scalar::Float64(v) => v.into_py(py),
                    Scalar::Utf8(v) => v.into_py(py),
                })
                .collect();
            d.set_item(c.name(), vals)?;
        }
        Ok(d)
    }

    /// Stable sort by one column or a list of columns, nulls last.
    /// `descending` is a bool or one bool per column.
    #[pyo3(signature = (by, descending = None))]
    fn sort(&self, by: &Bound<'_, PyAny>, descending: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let keys = str_or_list(by)?;
        let desc = match descending {
            Some(d) => bool_or_list(d, keys.len())?,
            None => vec![false; keys.len()],
        };
        let k: Vec<&str> = keys.iter().map(String::as_str).collect();
        Ok(PyDataFrame {
            df: self.df.sort_by_multi(&k, &desc).map_err(err)?,
        })
    }

    /// Start a group-by; finish with `.agg([(column, "sum"|"mean"|"min"|"max"|"count"), ...])`.
    fn group_by(&self, keys: Vec<String>) -> PyGroupBy {
        PyGroupBy {
            df: self.df.clone(),
            keys,
        }
    }

    /// Lazy query over this frame.
    fn lazy(&self) -> PyLazyFrame {
        PyLazyFrame {
            plan: LogicalPlan::scan(self.df.clone()),
        }
    }

    /// Zero-copy view of an Int32/Int64/Float64 column; `numpy.asarray(view)`
    /// shares the DataFrame's memory.
    fn column_view(&self, name: &str) -> PyResult<PyColumnView> {
        let s = self.df.column(name).map_err(err)?;
        let (data, nulls) = match s.array() {
            Array::Int64(a) => (ViewData::I64(a.values().clone()), a.null_count()),
            Array::Int32(a) => (ViewData::I32(a.values().clone()), a.null_count()),
            Array::Float64(a) => (ViewData::F64(a.values().clone()), a.null_count()),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "column {name:?} is not numeric; only Int32/Int64/Float64 export a buffer"
                )))
            }
        };
        Ok(PyColumnView {
            data,
            shape: s.len() as ffi::Py_ssize_t,
            nulls,
        })
    }

    fn __repr__(&self) -> String {
        self.df.to_string()
    }
}

/// Pending group-by.
#[pyclass(name = "GroupBy")]
struct PyGroupBy {
    df: DataFrame,
    keys: Vec<String>,
}

#[pymethods]
impl PyGroupBy {
    fn agg(&self, aggs: Vec<(String, String)>) -> PyResult<PyDataFrame> {
        let parsed = parse_aggs(aggs)?;
        let a: Vec<(&str, AggKind)> = parsed.iter().map(|(c, k)| (c.as_str(), *k)).collect();
        let keys: Vec<&str> = self.keys.iter().map(String::as_str).collect();
        Ok(PyDataFrame {
            df: self.df.group_by(&keys, &a).map_err(err)?,
        })
    }
}

/// A lazy query: transformations build a plan; `collect()` optimises and runs it.
#[pyclass(name = "LazyFrame")]
struct PyLazyFrame {
    plan: LogicalPlan,
}

#[pymethods]
impl PyLazyFrame {
    fn filter(&self, predicate: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyLazyFrame {
            plan: self.plan.clone().filter(to_expr(predicate)?),
        })
    }
    fn select(&self, columns: Vec<String>) -> Self {
        let names: Vec<&str> = columns.iter().map(String::as_str).collect();
        PyLazyFrame {
            plan: self.plan.clone().select(&names),
        }
    }
    fn with_columns(&self, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(PyLazyFrame {
            plan: self.plan.clone().with_columns(exprs_from(exprs)?),
        })
    }
    fn group_by(&self, keys: Vec<String>) -> PyLazyGroupBy {
        PyLazyGroupBy {
            plan: self.plan.clone(),
            keys,
        }
    }
    #[pyo3(signature = (other, on = None, how = "inner", left_on = None, right_on = None))]
    fn join(
        &self,
        other: &PyLazyFrame,
        on: Option<&Bound<'_, PyAny>>,
        how: &str,
        left_on: Option<&Bound<'_, PyAny>>,
        right_on: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (lk, rk) = join_key_args(on, left_on, right_on)?;
        let lk: Vec<&str> = lk.iter().map(String::as_str).collect();
        let rk: Vec<&str> = rk.iter().map(String::as_str).collect();
        Ok(PyLazyFrame {
            plan: self
                .plan
                .clone()
                .join_with(other.plan.clone(), &lk, &rk, parse_how(how)?),
        })
    }
    #[pyo3(signature = (by, descending = None))]
    fn sort(&self, by: &Bound<'_, PyAny>, descending: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let keys = str_or_list(by)?;
        let desc = match descending {
            Some(d) => bool_or_list(d, keys.len())?,
            None => vec![false; keys.len()],
        };
        let k: Vec<&str> = keys.iter().map(String::as_str).collect();
        Ok(PyLazyFrame {
            plan: self.plan.clone().sort(&k, &desc),
        })
    }
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> Self {
        PyLazyFrame {
            plan: self.plan.clone().limit(n),
        }
    }
    /// Optimise and execute.
    fn collect(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        PyDataFrame::run(py, self.plan.clone())
    }
    /// Unoptimised and optimised plans as text.
    fn explain(&self) -> String {
        self.plan.explain()
    }
    /// Output column names (a CSV scan only reads its header).
    #[getter]
    fn columns(&self) -> PyResult<Vec<String>> {
        self.plan.columns().map_err(err)
    }
    /// `{column: dtype}`; reads the file if the plan scans a CSV.
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (n, t) in self.plan.schema().map_err(err)? {
            d.set_item(n, format!("{t:?}").to_lowercase())?;
        }
        Ok(d)
    }
    fn __repr__(&self) -> String {
        self.plan.to_string()
    }
}

/// Pending lazy group-by.
#[pyclass(name = "LazyGroupBy")]
struct PyLazyGroupBy {
    plan: LogicalPlan,
    keys: Vec<String>,
}

#[pymethods]
impl PyLazyGroupBy {
    fn agg(&self, aggs: Vec<(String, String)>) -> PyResult<PyLazyFrame> {
        let parsed = parse_aggs(aggs)?;
        let a: Vec<(&str, AggKind)> = parsed.iter().map(|(c, k)| (c.as_str(), *k)).collect();
        let keys: Vec<&str> = self.keys.iter().map(String::as_str).collect();
        Ok(PyLazyFrame {
            plan: self.plan.clone().group_by(&keys, &a),
        })
    }
}

/// Read a CSV file into a DataFrame. `infer_rows` limits type inference to the
/// first N rows; `columns` parses only the named columns, in that order.
#[pyfunction]
#[pyo3(signature = (path, infer_rows = None, columns = None))]
fn read_csv(
    py: Python<'_>,
    path: &str,
    infer_rows: Option<usize>,
    columns: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let opts = csv_options(infer_rows, columns);
    let path = path.to_owned();
    let df = py
        .allow_threads(move || tessera_io::read_csv(path, &opts))
        .map_err(err)?;
    Ok(PyDataFrame { df })
}

/// Lazily scan a CSV file; the optimiser pushes the needed columns into the parser.
#[pyfunction]
#[pyo3(signature = (path, infer_rows = None))]
fn scan_csv(path: &str, infer_rows: Option<usize>) -> PyResult<PyLazyFrame> {
    Ok(PyLazyFrame {
        plan: LogicalPlan::scan_csv(path, csv_options(infer_rows, None)).map_err(err)?,
    })
}

#[pymodule]
fn tessera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyGroupBy>()?;
    m.add_class::<PyLazyFrame>()?;
    m.add_class::<PyLazyGroupBy>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyColumnView>()?;
    m.add_function(wrap_pyfunction!(py_col, m)?)?;
    m.add_function(wrap_pyfunction!(lit, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(scan_csv, m)?)?;
    Ok(())
}
