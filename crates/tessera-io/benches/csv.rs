//! CSV reader benchmarks: `cargo bench -p tessera-io --bench csv`.
//! Rows: 1M by default; set `TESSERA_CSV_ROWS=10000000` for 10M (needs ~1.5 GB).
use std::fmt::Write as _;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tessera_io::{read_csv_bytes_with, CsvOptions};

fn lcg(i: usize) -> u64 {
    (i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        >> 33
}

fn bench(c: &mut Criterion) {
    let n: usize = std::env::var("TESSERA_CSV_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let mut csv = String::with_capacity(n * 28);
    csv.push_str("id,key,value,name\n");
    for i in 0..n {
        let _ = writeln!(
            csv,
            "{i},{},{}.5,n{}",
            lcg(i) % 1000,
            lcg(i + 7) % 10_000,
            lcg(i) % 50
        );
    }
    let bytes = csv.as_bytes();
    let mut g = c.benchmark_group(format!("csv_{n}_rows"));
    g.throughput(Throughput::Bytes(bytes.len() as u64));
    g.sample_size(10)
        .measurement_time(Duration::from_secs(if n > 1_000_000 { 20 } else { 8 }));
    g.bench_function("parallel_all_columns", |b| {
        b.iter(|| read_csv_bytes_with(black_box(bytes), &CsvOptions::new()).unwrap())
    });
    g.bench_function("sequential_all_columns", |b| {
        b.iter(|| read_csv_bytes_with(black_box(bytes), &CsvOptions::new().with_chunks(1)).unwrap())
    });
    g.bench_function("parallel_projection_2_of_4", |b| {
        let o = CsvOptions::new().with_projection(["id", "value"]);
        b.iter(|| read_csv_bytes_with(black_box(bytes), &o).unwrap())
    });
    g.bench_function("parallel_infer_first_1000_rows", |b| {
        let o = CsvOptions::new().with_infer_rows(1000);
        b.iter(|| read_csv_bytes_with(black_box(bytes), &o).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
