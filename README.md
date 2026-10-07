<div align="center">

# tessera

A columnar DataFrame engine in Rust, built on a from-scratch implementation of the Apache Arrow memory layout. No arrow-rs, no polars underneath.

![License](https://img.shields.io/badge/LICENSE-MIT-blue?style=for-the-badge)  ![Language](https://img.shields.io/badge/RUST-dea584?style=for-the-badge&logo=rust&logoColor=white)  ![Tests](https://img.shields.io/badge/TESTS-107%20RUST%20%2B%20132%20PYTEST-brightgreen?style=for-the-badge)

Refer to [LICENSE](LICENSE) for the legal bits.

</div>

## 🏗️ Architecture

```
 +----------------------------+     +--------------------------------+
 | tessera-io                 |---->| tessera-core                   |
 |  read_csv_bytes (infer)    |     |  Series, DataFrame             |
 |  write_csv_string          |     |  select / filter (rayon) /     |
 +----------------------------+     |  head / group_by / sort / join |
                                    +---------------+----------------+
 +----------------------------+                     | uses
 | tessera-lazy               |---------------------+
 |  Expr, LogicalPlan,        |                     |
 |  optimiser, explain()      |     +---------------v----------------+
 +----------------------------+     | tessera-compute                |
                                    |  aggregate: sum / min / max    |
 +----------------------------+     |  filter, comparison, arithmetic|
 | tessera-py (pyo3)          |     |  scalar | chunked | AVX2       |
 +----------------------------+     +---------------+----------------+
                                                    | uses
                                    +---------------v----------------+
                                    | tessera-arrow                  |
                                    |  MutableBuffer -> Buffer       |
                                    |  Bitmap, Primitive / Boolean / |
                                    |  Utf8 arrays, builders, Array  |
                                    +--------------------------------+
```

* `tessera-arrow`: the memory layout. `MutableBuffer<T>` freezes into an `Arc`-backed `Buffer<T>` (with offset and length), `Bitmap` is LSB-ordered with a bit offset, and the array types are `PrimitiveArray`, `BooleanArray` and `Utf8Array` (i32 offsets plus a data buffer), with builders and an `Array` enum.
* `tessera-compute`: vectorised kernels. Sum/min/max, filter (word-at-a-time, plus an AVX2 compress-store using a LUT permute), comparisons that pack 64 lanes into bitmap words, and wrapping arithmetic. Each has a scalar path, a chunked path that leans on auto-vectorisation, and an AVX2 path used where the CPU has it.
* `tessera-core`: `Series` and `DataFrame` with select, parallel filter, head, `Display`, hash group-by, stable sort (single and multi-key) and inner/left/right/outer hash joins.
* `tessera-io`: CSV reader with schema inference and parallel chunked parsing, plus a writer.
* `tessera-lazy`: expression evaluation (vectorised where the kernels allow it, row-wise otherwise), logical plans (Scan, ScanCsv, Filter, Select, WithColumns, GroupBy, Join, Sort, Limit) with schema inference, and a rule-based optimiser: constant folding, filter merging, predicate pushdown and projection pushdown. `explain()` prints the plan before and after.
* `tessera-py`: the Python module. `DataFrame`, `Expr`, `LazyFrame`, `scan_csv`, and zero-copy NumPy export for numeric columns.

## 🚀 Quick start

```sh
cargo test --workspace --locked
cargo +nightly miri test -p tessera-arrow
cargo bench -p tessera-compute --bench kernels
```

The Python side needs maturin and a venv:

```sh
python3 -m venv .venv && .venv/bin/pip install maturin pytest numpy
(cd crates/tessera-py && VIRTUAL_ENV=$PWD/../../.venv PATH=$PWD/../../.venv/bin:$PATH maturin develop)
.venv/bin/pip install pandas && .venv/bin/pytest python/tests
.venv/bin/pip install polars && .venv/bin/python python/bench.py   # optional comparison
```

Fuzzing uses cargo-fuzz on nightly, for example `cargo +nightly fuzz run csv_parse -- -max_total_time=60`.

A quick taste:

```rust
use tessera_core::AggKind;
let df = tessera_io::read_csv_bytes(b"k,v\na,1\nb,\na,3\n")?;
let g = df.group_by(&["k"], &[("v", AggKind::Sum), ("v", AggKind::Count)])?;
println!("{g}");
```

which prints (`cargo run -p tessera-io --example groupby`):

```
shape: (2, 3)
+-----+-------+---------+
| k   | v_sum | v_count |
| str | i64   | i64     |
+-----+-------+---------+
| "a" | 4     | 2       |
| "b" | null  | 0       |
+-----+-------+---------+
```

## 🧪 Testing

107 Rust tests and 132 pytest cases, all passing.

* `tessera-arrow` (12): buffers, bitmaps, offsets and builders. These also run clean under Miri.
* `tessera-compute` (31 unit, 13 differential): every kernel is compared between its scalar, chunked and AVX2 paths.
* `tessera-core` and `tessera-io`: group-by, join variants, sort, CSV round trips.
* `tessera-lazy`: proptests checking that optimised and unoptimised plans give identical results over every plan node, that kernel evaluation matches row-wise evaluation (comparisons, `+ - *`, AND/OR trees, and that the kernel path was actually taken), and that random whole pipelines (filter, with_columns, select, join, sort, group-by, limit; 400 cases) match a naive row-based reference.
* `python/tests`: group-by, sort, join (all four kinds, `on` and `left_on`/`right_on`, eager and lazy) and filter compared against pandas, `read_csv`, expression operators, lazy `scan_csv`, multi-key sort, and the NumPy zero-copy export.
* Fuzz targets: `csv_parse` and `expr_eval`. The latter also pushes each expression through join/filter/sort/group-by/limit and asserts optimised equals unoptimised.
* AddressSanitizer is clean on `tessera-arrow` and on `tessera-compute` (the AVX2 unsafe code). I checked with a deliberate out-of-bounds canary that the sanitiser really was active.

## ⏱️ Benchmarks

These are from a shared WSL2 box with 28 logical cores and other work running, so treat anything under about 20% as noise. All are single runs.

Kernels (criterion, 4M elements, AVX2 available):

| Kernel | scalar | chunked | AVX2 |
|---|---|---|---|
| `sum_f64` | 2.26 ms | 1.24 ms | 1.19 ms |
| `filter_u64`, 50% random mask | 4.66 ms | 4.96 ms | 3.95 ms |

Once the scalar sum is split into independent accumulators, LLVM gets within about 5% of the hand-written AVX2 (it's memory-bound). The word-at-a-time filter doesn't beat the naive branchy one on a random 50% mask; the AVX2 compress-store is about 20% faster.

Core operators (`cargo bench -p tessera-io --bench ops`, median of 10):

| operation | 1M rows | 10M rows |
|---|---|---|
| `group_by([name, key])` sum+mean, ~50k groups | 13.1 ms | 164 ms |
| `group_by([name, hc])`, 100k distinct | 30.7 ms | 201 ms |
| `group_by([key])`, Int64 fast path | 4.9 ms | 65.9 ms |
| `sort_by_multi([name, key])`, all columns gathered | 91.4 ms | 1.63 s |
| `join_on` 1M x 1k | 27.4 ms | 314 ms |

CSV reader (`cargo bench -p tessera-io --bench csv`):

| variant | 1M rows (~27 MB) | 10M rows (~290 MB) |
|---|---|---|
| parallel, all 4 columns | 18.0 ms | 174.6 ms |
| sequential | 46.9 ms | 603.5 ms |
| parallel, 2 of 4 columns | 11.4 ms | 135.2 ms |

Python versus pandas 3.1.1 and polars 2.1.0 (`python/bench.py`, 1M rows, best of 3, ms):

| operation | tessera | pandas | polars |
|---|---|---|---|
| filter `v > 0` | 4.2 | 3.3 | 3.1 |
| group_by `k` (1000 groups) sum, mean | 5.5 | 11.4 | 8.2 |
| sort by `v` (stable) | 40.4 | 72.0 | 22.3 |
| inner join 1M x 100k | 11.6 | 17.4 | 7.4 |
| left join 1M x 100k | 17.9 | 12.7 | 10.0 |
| read_csv 1M x 4 | 27.7 | 32.5 | 12.0 |
| read_csv, only `id`,`x` | 18.4 | 37.9 | 10.0 |
| lazy `scan_csv.filter.select.collect` | 19.5 | 35.8 | 10.5 |

tessera beats pandas on group-by, sort, inner join and the projected/lazy CSV reads, and loses to polars on nearly everything. The left join is slower than pandas because right columns are gathered through a nullable `take_opt` that I haven't optimised. Pinned to 4 threads the picture is similar, except the full CSV read also falls behind pandas.

## ⚠️ Honest limits

* Only `Boolean`, `Int32`, `Int64`, `Float64` and `Utf8`. No nested types, IPC or C Data Interface.
* Row ids are `u32`, so frames are limited to `u32::MAX` rows.
* The radix/argsort-based sort is single-threaded, and so is the join build side (probing is parallel, as is the matched part of right/outer joins; the unmatched-right pass is not).
* Group-by on a single Int64 key uses a single-threaded `hashbrown` fast path. Other keys use a partitioned parallel path, but the group-id scatter and per-column aggregation passes aren't partition-parallel.
* Lazy layer: no string functions and no aggregation expressions (`GroupBy` takes `(column, AggKind)` pairs). No join reordering. There's no physical plan layer, so logical nodes execute directly. Expression evaluation isn't parallel. Utf8 expressions other than comparisons fall back to row-wise evaluation or error.
* Predicates are pushed below `GroupBy` (key-only conjuncts) and `WithColumns`, never below `Limit`, and only into the side of a join that keeps all its rows. CSV scans have no pushed-down predicate.
* CSV is chunk-parallel only for quote-free input; quoted input is a single chunk. The whole input is read into memory, and with `infer_rows` a later field that doesn't fit the inferred type is an error rather than null.
* Float `min`/`max` ignore NaN (pandas `skipna` semantics) and integer arithmetic wraps.
* Python: `column_view` exports Int32/Int64/Float64 columns zero-copy and read-only, but there's no Boolean/Utf8 export, no building a DataFrame from NumPy arrays, and no pandas/Arrow interchange. `to_dict` materialises Python objects.
* No wheels, abi3 or PyPI packaging; you build it with maturin.
* Miri only runs on `tessera-arrow`, and AddressSanitizer on `tessera-arrow` and `tessera-compute`. No ThreadSanitizer run, and the Python extension hasn't been run under a sanitiser.

## Licence

Licensed under either of Apache License 2.0 or MIT, at your option.
