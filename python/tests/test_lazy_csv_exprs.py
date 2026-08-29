"""read_csv / scan_csv, with_columns, head, Expr operators and lazy().collect()."""
import math
import random

import pandas as pd
import pytest

import tessera
from tessera import col, lit


@pytest.fixture()
def csv_path(tmp_path):
    rng = random.Random(7)
    rows = ["id,grp,x,y,name"]
    for i in range(400):
        x = "" if rng.random() < 0.1 else str(rng.randrange(-50, 50))
        y = "" if rng.random() < 0.1 else f"{rng.uniform(-3, 3):.3f}"
        rows.append(f"{i},{rng.choice('abcd')},{x},{y},n{rng.randrange(20)}")
    p = tmp_path / "t.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def _norm(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def test_read_csv_matches_pandas(csv_path):
    got = tessera.read_csv(csv_path)
    exp = pd.read_csv(csv_path)
    assert got.shape == exp.shape
    assert got.columns == list(exp.columns)
    d = got.to_dict()
    for c in exp.columns:
        assert [_norm(v) for v in d[c]] == [_norm(v) for v in exp[c].tolist()], c


def test_read_csv_projection_and_infer_rows(csv_path):
    got = tessera.read_csv(csv_path, columns=["y", "id"], infer_rows=100)
    assert got.columns == ["y", "id"]
    assert got.shape == (400, 2)
    with pytest.raises(ValueError):
        tessera.read_csv(csv_path, columns=["nope"])
    with pytest.raises(ValueError):
        tessera.read_csv(csv_path + ".missing")


def test_head_and_len():
    df = tessera.DataFrame({"a": list(range(10))})
    assert df.head(3).to_dict() == {"a": [0, 1, 2]}
    assert len(df.head(100)) == 10
    assert df.head().shape == (5, 1)


def test_expr_operators_and_repr():
    e = (((col("a") + 1) * 2 - col("b")) > 3) & (col("a") == 0)
    assert "AND" in repr(e) and "col(a)" in repr(e)
    assert repr(1 + col("a")) == "(1 + col(a))"
    assert repr(10 - col("a")) == "(10 - col(a))"
    assert repr(col("a") | (col("b") < lit(2.5)))== "(col(a) OR (col(b) < 2.5))"
    assert repr(col("a").alias("z")).endswith("AS z")
    with pytest.raises(TypeError):
        bool(col("a") > 1)
    with pytest.raises(TypeError):
        col("a") + object()
    assert repr(col("a") >= 1) == "(col(a) >= 1)"
    assert repr(col("a") <= 1) == "(col(a) <= 1)"
    assert repr(col("a") != 1) == "(col(a) != 1)"
    assert repr(~(col("a") > 1)) == "NOT (col(a) > 1)"


def test_with_columns_matches_python():
    a = [1, None, 3, 4]
    b = [10, 20, 30, None]
    df = tessera.DataFrame({"a": a, "b": b, "f": [0.5, 1.5, None, 2.0]})
    out = df.with_columns(
        [
            (col("a") + col("b")).alias("s"),
            (col("a") * 2 - 1).alias("t"),
            (col("f") * col("a")).alias("p"),
            (col("a") > 2).alias("big"),
            col("b").alias("a"),  # replaces a in place
        ]
    ).to_dict()
    assert list(out) == ["a", "b", "f", "s", "t", "p", "big"]
    assert out["a"] == b
    assert out["s"] == [None if x is None or y is None else x + y for x, y in zip(a, b)]
    assert out["t"] == [None if x is None else x * 2 - 1 for x in a]
    assert out["p"] == [0.5, None, None, 8.0]
    assert out["big"] == [False, None, True, True]


def test_lazy_collect_equals_eager(csv_path):
    eager = tessera.read_csv(csv_path)
    lazy = (
        eager.lazy()
        .filter((col("x") > -20) & (col("id") < 300))
        .with_columns([(col("x") * 2).alias("x2")])
        .group_by(["grp"])
        .agg([("x2", "sum"), ("id", "count")])
        .sort("grp")
        .collect()
    )
    pdf = pd.read_csv(csv_path)
    pdf = pdf[(pdf.x > -20) & (pdf.id < 300)].assign(x2=lambda d: d.x * 2)
    exp = pdf.groupby("grp").agg(x2_sum=("x2", "sum"), id_count=("id", "count")).reset_index().sort_values("grp")
    got = lazy.to_dict()
    assert got["grp"] == exp["grp"].tolist()
    assert got["x2_sum"] == [int(v) for v in exp["x2_sum"]]
    assert got["id_count"] == exp["id_count"].tolist()


def test_lazy_join_head_and_explain():
    l = tessera.DataFrame({"k": [1, 2, 3, 4], "a": [10, 20, 30, 40]}).lazy()
    r = tessera.DataFrame({"k": [2, 3, 5], "b": [200, 300, 500]}).lazy()
    q = l.join(r, "k", how="left").filter(col("a") > 10).sort("a", descending=True).head(2)
    out = q.collect().to_dict()
    assert out == {"k": [4, 3], "a": [40, 30], "b": [None, 300]}
    plan = q.explain()
    assert "== optimized ==" in plan
    # the predicate on the left-only column is pushed into the left scan
    assert "filter=(col(a) > 10)" in plan.split("== optimized ==")[1]
    assert q.columns == ["k", "a", "b"]
    assert q.schema() == {"k": "int64", "a": "int64", "b": "int64"}


def test_scan_csv_projection_pushdown(csv_path):
    q = tessera.scan_csv(csv_path).filter(col("x") > 0).select(["id", "x"])
    assert q.columns == ["id", "x"]
    optimized = q.explain().split("== optimized ==")[1]
    assert "SCAN CSV" in optimized and "[id, x]" in optimized
    got = q.collect().to_dict()
    pdf = pd.read_csv(csv_path)
    exp = pdf[pdf.x > 0]
    assert got["id"] == exp["id"].tolist()
    assert got["x"] == exp["x"].astype(int).tolist()
    assert q.schema() == {"id": "int64", "x": "int64"}


def test_multi_key_sort():
    df = tessera.DataFrame({"a": [2, 1, 2, 1, None], "b": [1, 2, 3, 4, 5]})
    out = df.sort(["a", "b"], descending=[False, True]).to_dict()
    assert out["a"] == [1, 1, 2, 2, None]
    assert out["b"] == [4, 2, 3, 1, 5]
    with pytest.raises(ValueError):
        df.sort(["a", "b"], descending=[True])


def test_ge_le_ne_not_match_pandas(csv_path):
    pdf = pd.read_csv(csv_path)
    df = tessera.read_csv(csv_path)
    for name, e, mask in [
        ("ge", col("x") >= 3, pdf.x >= 3),
        ("le", col("y") <= 0.5, pdf.y <= 0.5),
        ("ne", col("x") != 7, pdf.x != 7),
        ("ne_str", col("grp") != "a", pdf.grp != "a"),
        ("ge_cols", col("x") >= col("id"), pdf.x >= pdf.id),
        ("not", ~(col("id") >= 100), ~(pdf.id >= 100)),
    ]:
        got = df.lazy().filter(e).collect().to_dict()["id"]
        # null comparisons are null in tessera (row dropped); pandas gives
        # False for >= <= and True for !=, so drop null-operand rows from pandas.
        used = [c for c in ("x", "y", "id", "grp") if f"col({c})" in repr(e)]
        valid = pdf[used].notna().all(axis=1)
        assert got == pdf[mask & valid].id.tolist(), name
