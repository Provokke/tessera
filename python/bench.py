"""Benchmark: tessera vs pandas (and polars when installed) for filter,
group-by, sort, join and read_csv.

Run after `maturin develop --release`: .venv/bin/python python/bench.py
Construction from Python lists is excluded from timings; each op is the
best of 3 runs, in milliseconds. Results are not converted back to Python.
"""
import os
import random
import tempfile
import time

import pandas as pd

import tessera

try:
    import polars as pl
except ImportError:  # polars is optional
    pl = None

N = 1_000_000
rng = random.Random(0)
data = {"k": [rng.randrange(1000) for _ in range(N)], "v": [rng.randrange(-10_000, 10_000) for _ in range(N)]}
tdf = tessera.DataFrame(data)
pdf = pd.DataFrame(data)
pldf = pl.DataFrame(data) if pl else None

# join: 1M-row left, 100k-row right with unique keys
left = {"k": [rng.randrange(150_000) for _ in range(N)], "v": data["v"]}
right = {"k": list(range(100_000)), "w": [rng.randrange(1000) for _ in range(100_000)]}
tl, tr = tessera.DataFrame(left), tessera.DataFrame(right)
pl_l, pl_r = pd.DataFrame(left), pd.DataFrame(right)
pol_l, pol_r = (pl.DataFrame(left), pl.DataFrame(right)) if pl else (None, None)

# csv: 1M rows, int / int / float / short string
csv_path = os.path.join(tempfile.gettempdir(), "tessera_bench.csv")
with open(csv_path, "w") as f:
    f.write("id,k,x,name\n")
    for i in range(N):
        f.write(f"{i},{rng.randrange(1000)},{rng.uniform(-1, 1):.6f},n{rng.randrange(100)}\n")


def best(fn, reps=3):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times) * 1000


def row(name, t, p, o):
    return (name, best(t), best(p), best(o) if (pl and o) else None)


rows = [
    row("filter v > 0", lambda: tdf.filter("v", ">", 0), lambda: pdf[pdf["v"] > 0],
        lambda: pldf.filter(pl.col("v") > 0) if pl else None),
    row("group_by k: sum, mean", lambda: tdf.group_by(["k"]).agg([("v", "sum"), ("v", "mean")]),
        lambda: pdf.groupby("k", sort=False).agg(s=("v", "sum"), m=("v", "mean")),
        lambda: pldf.group_by("k").agg(pl.col("v").sum().alias("s"), pl.col("v").mean().alias("m")) if pl else None),
    row("sort by v", lambda: tdf.sort("v"), lambda: pdf.sort_values("v", kind="stable"),
        lambda: pldf.sort("v", maintain_order=True) if pl else None),
    row("inner join 1M x 100k on k", lambda: tl.join(tr, "k"), lambda: pl_l.merge(pl_r, on="k"),
        lambda: pol_l.join(pol_r, on="k", how="inner") if pl else None),
    row("left join 1M x 100k on k", lambda: tl.join(tr, "k", how="left"), lambda: pl_l.merge(pl_r, on="k", how="left"),
        lambda: pol_l.join(pol_r, on="k", how="left") if pl else None),
    row("read_csv 1M x 4", lambda: tessera.read_csv(csv_path), lambda: pd.read_csv(csv_path),
        lambda: pl.read_csv(csv_path) if pl else None),
    row("read_csv 1M, columns id,x only", lambda: tessera.read_csv(csv_path, columns=["id", "x"]),
        lambda: pd.read_csv(csv_path, usecols=["id", "x"]),
        lambda: pl.read_csv(csv_path, columns=["id", "x"]) if pl else None),
    row("lazy: scan_csv.filter(k>500).select(id,k).collect",
        lambda: tessera.scan_csv(csv_path).filter(tessera.col("k") > 500).select(["id", "k"]).collect(),
        lambda: (lambda d: d[d["k"] > 500])(pd.read_csv(csv_path, usecols=["id", "k"])),
        lambda: pl.scan_csv(csv_path).filter(pl.col("k") > 500).select(["id", "k"]).collect() if pl else None),
]
print(f"N = {N:,} rows; pandas {pd.__version__}" + (f", polars {pl.__version__}" if pl else ", polars not installed"))
print("| operation | tessera (ms) | pandas (ms) | polars (ms) |")
print("|---|---|---|---|")
for name, t, p, o in rows:
    print(f"| {name} | {t:.1f} | {p:.1f} | {'n/a' if o is None else f'{o:.1f}'} |")
os.remove(csv_path)
