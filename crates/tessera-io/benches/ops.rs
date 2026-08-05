//! End-to-end operator benchmarks on 1M rows: CSV read, group-by, sort, join.
use std::fmt::Write as _;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tessera_arrow::Utf8Array;
use tessera_arrow::{Array, PrimitiveArray};
use tessera_core::{AggKind, DataFrame, JoinType, Series};

const N: usize = 1_000_000;

fn lcg(i: usize) -> u64 {
    (i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        >> 33
}

fn bench(c: &mut Criterion) {
    let mut csv = String::from("id,key,value,name\n");
    for i in 0..N {
        let _ = writeln!(
            csv,
            "{i},{},{}.5,n{}",
            lcg(i) % 1000,
            lcg(i + 7) % 10_000,
            lcg(i) % 50
        );
    }
    let mut g = c.benchmark_group("ops_1m_rows");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("csv_read", |b| {
        b.iter(|| tessera_io::read_csv_bytes(black_box(csv.as_bytes())).unwrap())
    });

    let df = tessera_io::read_csv_bytes(csv.as_bytes()).unwrap();
    g.bench_function("group_by_1k_groups_sum_mean", |b| {
        b.iter(|| {
            df.group_by(
                &["key"],
                &[("value", AggKind::Sum), ("value", AggKind::Mean)],
            )
            .unwrap()
        })
    });
    g.bench_function("sort_by_f64", |b| {
        b.iter(|| df.sort_by("value", false).unwrap())
    });

    let dim = DataFrame::new(vec![
        Series::new(
            "key",
            Array::Int64(PrimitiveArray::from_options((0..1000i64).map(Some))),
        ),
        Series::new(
            "w",
            Array::Int64(PrimitiveArray::from_options(
                (0..1000i64).map(|i| Some(i * 2)),
            )),
        ),
    ])
    .unwrap();
    g.bench_function("inner_join_1m_x_1k", |b| {
        b.iter(|| df.join(&dim, "key").unwrap())
    });
    g.finish();
}

/// Group-by / sort / join on generated frames. Rows: `TESSERA_OPS_ROWS` (default 1M).
fn bench_core(c: &mut Criterion) {
    let n: usize = std::env::var("TESSERA_OPS_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(N);
    let names: Vec<String> = (0..50).map(|i| format!("n{i}")).collect();
    let df = DataFrame::new(vec![
        Series::new(
            "name",
            Array::Utf8(
                Utf8Array::from_options(
                    (0..n).map(|i| Some(names[(lcg(i) % 50) as usize].as_str())),
                )
                .unwrap(),
            ),
        ),
        Series::new(
            "key",
            Array::Int64(PrimitiveArray::from_options(
                (0..n).map(|i| Some((lcg(i + 1) % 1000) as i64)),
            )),
        ),
        Series::new(
            "hc",
            Array::Int64(PrimitiveArray::from_options(
                (0..n).map(|i| Some((lcg(i + 2) % 100_000) as i64)),
            )),
        ),
        Series::new(
            "value",
            Array::Float64(PrimitiveArray::from_options(
                (0..n).map(|i| Some((lcg(i + 3) % 10_000) as f64 + 0.5)),
            )),
        ),
    ])
    .unwrap();
    let aggs = [("value", AggKind::Sum), ("value", AggKind::Mean)];
    let mut g = c.benchmark_group(format!("core_{n}_rows"));
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("group_by_name_key_50k_groups_partitioned", |b| {
        b.iter(|| df.group_by(&["name", "key"], &aggs).unwrap())
    });
    g.bench_function("group_by_name_key_50k_groups_reference", |b| {
        b.iter(|| df.group_by_reference(&["name", "key"], &aggs).unwrap())
    });
    g.bench_function("group_by_name_hc_high_cardinality_partitioned", |b| {
        b.iter(|| df.group_by(&["name", "hc"], &aggs).unwrap())
    });
    g.bench_function("group_by_key_int64_fast_path", |b| {
        b.iter(|| df.group_by(&["key"], &aggs).unwrap())
    });
    g.bench_function("sort_by_name_key_desc", |b| {
        b.iter(|| df.sort_by_multi(&["name", "key"], &[false, true]).unwrap())
    });
    let dim = df
        .head(1000.min(n))
        .unwrap()
        .select(&["name", "key", "value"])
        .unwrap();
    g.bench_function("left_join_on_name_key", |b| {
        b.iter(|| df.join_on(&dim, &["name", "key"], JoinType::Left).unwrap())
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(3)).warm_up_time(Duration::from_millis(300));
    targets = bench, bench_core
}
criterion_main!(benches);
