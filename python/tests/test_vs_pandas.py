import math
import random

import pandas as pd
import pytest

import tessera


def _data(n=500, seed=0):
    rng = random.Random(seed)
    return {
        "k": [rng.choice(["a", "b", "c", None]) for _ in range(n)],
        "v": [rng.choice([None, *range(-20, 20)]) for _ in range(n)],
        "f": [rng.choice([None, rng.uniform(-5, 5)]) for _ in range(n)],
    }


def _norm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


@pytest.mark.parametrize("seed", range(5))
def test_group_by_matches_pandas(seed):
    data = _data(seed=seed)
    got = tessera.DataFrame(data).group_by(["k"]).agg(
        [("v", "sum"), ("v", "min"), ("v", "max"), ("v", "count"), ("f", "mean")]
    ).to_dict()
    pdf = pd.DataFrame(data).astype({"v": "Int64"})
    exp = pdf.groupby("k", dropna=False, sort=False).agg(
        v_sum=("v", "sum"), v_min=("v", "min"), v_max=("v", "max"), v_count=("v", "count"), f_mean=("f", "mean")
    )
    got_rows = {got["k"][i]: i for i in range(len(got["k"]))}
    assert len(got_rows) == len(exp)
    for key, row in exp.iterrows():
        i = got_rows[None if key is None or (isinstance(key, float) and math.isnan(key)) else key]
        # pandas sums an all-null group to 0; tessera returns null (documented difference).
        exp_sum = _norm(row["v_sum"]) if row["v_count"] > 0 else None
        assert got["v_sum"][i] == exp_sum
        assert got["v_min"][i] == _norm(row["v_min"])
        assert got["v_max"][i] == _norm(row["v_max"])
        assert got["v_count"][i] == row["v_count"]
        m = _norm(row["f_mean"])
        assert (got["f_mean"][i] is None and m is None) or got["f_mean"][i] == pytest.approx(m)


@pytest.mark.parametrize("desc", [False, True])
def test_sort_matches_pandas(desc):
    data = _data(seed=42)
    got = tessera.DataFrame(data).sort("v", descending=desc).to_dict()
    exp = pd.DataFrame(data).astype({"v": "Int64"}).sort_values("v", ascending=not desc, kind="stable", na_position="last")
    assert got["v"] == [_norm(x) if x is not pd.NA else None for x in exp["v"].tolist()]
    assert got["k"] == [None if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in exp["k"].tolist()]
