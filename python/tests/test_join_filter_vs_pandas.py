"""Join and filter results compared with pandas on random data with nulls."""
import math
import random

import pandas as pd
import pytest

import tessera
from tessera import col, lit


def _norm(x):
    if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _rows(columns: dict, order):
    n = len(next(iter(columns.values())))
    return sorted(
        (tuple(_norm(columns[c][i]) for c in order) for i in range(n)),
        key=lambda r: tuple((v is None, v if v is not None else 0) for v in r),
    )


def _frames(seed, nl=300, nr=120):
    rng = random.Random(seed)
    left = {
        "k": [rng.randrange(0, 40) for _ in range(nl)],
        "j": [rng.choice(["x", "y", "z"]) for _ in range(nl)],
        "v": [rng.choice([None, *range(-50, 50)]) for _ in range(nl)],
    }
    right = {
        "k": [rng.randrange(10, 60) for _ in range(nr)],
        "j": [rng.choice(["x", "y", "w"]) for _ in range(nr)],
        "w": [rng.choice([None, *range(100, 200)]) for _ in range(nr)],
        "v": [rng.randrange(1000, 1010) for _ in range(nr)],  # clashes with left "v"
    }
    return left, right


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("on", ["k", ["k", "j"]])
def test_join_matches_pandas(seed, how, on):
    left, right = _frames(seed)
    got = tessera.DataFrame(left).join(tessera.DataFrame(right), on, how=how).to_dict()
    keys = [on] if isinstance(on, str) else on
    exp = pd.DataFrame(left).astype({"v": "Int64"}).merge(
        pd.DataFrame(right).astype({"w": "Int64"}), on=keys, how=how, suffixes=("", "_right")
    )
    cols = list(exp.columns)
    assert list(got.keys()) == cols
    assert len(got["k"]) == len(exp)
    exp_dict = {c: exp[c].tolist() for c in cols}
    assert _rows(got, cols) == _rows(exp_dict, cols)


def test_left_join_keeps_left_order_and_null_keys_never_match():
    left = tessera.DataFrame({"k": [3, None, 1, 3], "a": [10, 20, 30, 40]})
    right = tessera.DataFrame({"k": [None, 3, 3], "b": [1, 2, 3]})
    got = left.join(right, "k", how="left").to_dict()
    assert got["k"] == [3, 3, None, 1, 3, 3]
    assert got["a"] == [10, 10, 20, 30, 40, 40]
    assert got["b"] == [2, 3, None, None, 2, 3]
    inner = left.join(right, "k").to_dict()
    assert inner["a"] == [10, 10, 40, 40]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("keys", [("k", "rk"), (["k", "j"], ["rk", "rj"])])
def test_join_left_on_right_on_matches_pandas(seed, how, lazy, keys):
    left, right = _frames(seed)
    right = {("rk" if c == "k" else "rj" if c == "j" else c): v for c, v in right.items()}
    left_on, right_on = keys
    ldf, rdf = tessera.DataFrame(left), tessera.DataFrame(right)
    if lazy:
        got = ldf.lazy().join(rdf.lazy(), how=how, left_on=left_on, right_on=right_on).collect()
    else:
        got = ldf.join(rdf, how=how, left_on=left_on, right_on=right_on)
    got = got.to_dict()
    exp = pd.DataFrame(left).astype({"v": "Int64"}).merge(
        pd.DataFrame(right).astype({"w": "Int64"}),
        left_on=left_on, right_on=right_on, how=how, suffixes=("", "_right"),
    )
    cols = list(exp.columns)
    assert list(got.keys()) == cols
    assert len(got["k"]) == len(exp)
    exp_dict = {c: exp[c].tolist() for c in cols}
    assert _rows(got, cols) == _rows(exp_dict, cols)


def test_right_and_outer_join_fill_the_key_and_keep_null_keys():
    left = tessera.DataFrame({"k": [1, None], "a": [10, 20]})
    right = tessera.DataFrame({"k": [None, 1, 2], "b": [1, 2, 3]})
    got = left.join(right, "k", how="right").to_dict()
    assert got["k"] == [1, None, 2]
    assert got["a"] == [10, None, None]
    assert got["b"] == [2, 1, 3]
    out = left.join(right, "k", how="outer").to_dict()
    assert out["k"] == [1, None, None, 2]
    assert out["a"] == [10, 20, None, None]
    assert out["b"] == [2, None, 1, 3]


def test_join_errors():
    a = tessera.DataFrame({"k": [1]})
    with pytest.raises(ValueError):
        a.join(a, "missing")
    with pytest.raises(ValueError):
        a.join(a, "k", how="cross")
    with pytest.raises(ValueError):
        a.join(a)  # no keys
    with pytest.raises(ValueError):
        a.join(a, "k", left_on="k", right_on="k")  # both forms
    with pytest.raises(ValueError):
        a.join(a, left_on="k")  # right_on missing
    with pytest.raises(ValueError):
        a.join(a, left_on=["k", "k"], right_on=["k"])  # lengths differ
    s = tessera.DataFrame({"k": ["x"]})
    with pytest.raises(ValueError):
        a.join(s, "k", how="outer")  # key types cannot be merged


def _pd_frame(seed, n=500):
    rng = random.Random(seed)
    data = {
        "a": [rng.choice([None, *range(-20, 20)]) for _ in range(n)],
        "b": [rng.randrange(-5, 5) for _ in range(n)],
        "f": [rng.choice([None, rng.uniform(-5, 5)]) for _ in range(n)],
    }
    return data, pd.DataFrame(data).astype({"a": "Int64"})


@pytest.mark.parametrize("seed", range(5))
def test_legacy_filter_matches_pandas(seed):
    data, pdf = _pd_frame(seed)
    t = tessera.DataFrame(data)
    for op, mask in [(">", pdf["a"] > 3), ("<", pdf["a"] < 3), ("==", pdf["a"] == 3)]:
        got = t.filter("a", op, 3).to_dict()
        exp = pdf[mask.fillna(False).astype(bool)]
        assert got["a"] == [_norm(x) for x in exp["a"]]
        assert got["b"] == exp["b"].tolist()


@pytest.mark.parametrize("seed", range(5))
def test_expr_filter_matches_pandas(seed):
    data, pdf = _pd_frame(seed)
    t = tessera.DataFrame(data)
    cases = [
        ((col("a") > 3) & (col("b") < 2), (pdf["a"] > 3) & (pdf["b"] < 2)),
        ((col("a") < col("b")) | (col("b") == 0), (pdf["a"] < pdf["b"]) | (pdf["b"] == 0)),
        ((col("a") + col("b")) * 2 > 10, (pdf["a"] + pdf["b"]) * 2 > 10),
        (col("f") > 0.5, pdf["f"] > 0.5),
        (col("f") > col("a"), pdf["f"].astype("Float64") > pdf["a"]),
        (5 < col("b"), pdf["b"] > 5),
    ]
    for expr, mask in cases:
        got = t.filter(expr).to_dict()
        # pandas three-valued OR keeps "NA or True"; tessera nulls propagate, so
        # compare against rows where the pandas mask is definitely true AND no
        # operand is null for the OR case.
        exp_mask = mask.fillna(False).astype(bool)
        if "|" in repr(expr) or "OR" in repr(expr):
            exp_mask = exp_mask & pdf["a"].notna()
        exp = pdf[exp_mask]
        assert got["b"] == exp["b"].tolist(), repr(expr)
        assert len(got["a"]) == len(exp), repr(expr)
