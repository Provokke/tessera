//! Kernel throughput: scalar vs auto-vectorized vs AVX2.
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tessera_arrow::{Bitmap, BooleanArray, PrimitiveArray, Utf8Array};
use tessera_compute::{
    aggregate, arithmetic, boolean, comparison, filter, hash, sort, strings, take,
};

fn bench(c: &mut Criterion) {
    let n = 4_000_000usize;
    let f: Vec<f64> = (0..n).map(|i| (i % 1000) as f64).collect();
    let mut g = c.benchmark_group("sum_f64");
    g.throughput(Throughput::Bytes((n * 8) as u64));
    g.bench_function("scalar", |b| {
        b.iter(|| aggregate::sum_f64_scalar(black_box(&f)))
    });
    g.bench_function("chunked", |b| {
        b.iter(|| aggregate::sum_f64_chunked(black_box(&f)))
    });
    g.bench_function("avx2", |b| {
        b.iter(|| aggregate::sum_f64_avx2(black_box(&f)))
    });
    g.finish();

    let u: Vec<u64> = (0..n as u64).collect();
    let sel = Bitmap::from_bools((0..n).map(|i| (i * 7919) % 100 < 50));
    let mut g = c.benchmark_group("filter_u64_50pct");
    g.throughput(Throughput::Bytes((n * 8) as u64));
    g.bench_function("scalar", |b| {
        b.iter(|| filter::filter_slice_scalar(black_box(&u), &sel))
    });
    g.bench_function("chunked", |b| {
        b.iter(|| filter::filter_slice_chunked(black_box(&u), &sel))
    });
    g.bench_function("avx2", |b| {
        b.iter(|| filter::filter_u64_avx2(black_box(&u), &sel))
    });
    g.finish();

    let a = BooleanArray::try_new(Bitmap::from_bools((0..n).map(|i| i % 3 == 0)), None).ok();
    let b = BooleanArray::try_new(Bitmap::from_bools((0..n).map(|i| i % 5 == 0)), None).ok();
    if let (Some(a), Some(b)) = (a, b) {
        let mut g = c.benchmark_group("boolean");
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("and", |bn| bn.iter(|| boolean::and(black_box(&a), &b)));
        g.bench_function("not", |bn| bn.iter(|| boolean::not(black_box(&a))));
        g.finish();
    }

    let i: PrimitiveArray<i64> =
        PrimitiveArray::from_options((0..n as i64).map(|x| Some(x ^ 0x5555)));
    let mut g = c.benchmark_group("minmax_i64");
    g.throughput(Throughput::Bytes((n * 8) as u64));
    g.bench_function("min", |bn| bn.iter(|| aggregate::min_i64(black_box(&i))));
    g.bench_function("max", |bn| bn.iter(|| aggregate::max_i64(black_box(&i))));
    g.finish();

    let idx: Vec<Option<u32>> = (0..n as u32)
        .map(|k| Some(k.wrapping_mul(2654435761) % n as u32))
        .collect();
    let mut g = c.benchmark_group("take_i64");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("random", |bn| {
        bn.iter(|| take::take_primitive(black_box(&i), &idx))
    });
    g.finish();

    let m = 1_000_000usize;
    let keys: Vec<Option<i64>> = (0..m as u64)
        .map(|k| Some((k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) as i64))
        .collect();
    let ka = PrimitiveArray::from_options(keys);
    let fa = PrimitiveArray::from_options((0..m).map(|k| Some(((k * 7919) % 100_003) as f64)));
    let mut g = c.benchmark_group("argsort_1m");
    g.throughput(Throughput::Elements(m as u64));
    g.bench_function("i64_radix", |bn| {
        bn.iter(|| sort::argsort_i64(black_box(&ka), false))
    });
    g.bench_function("f64_radix", |bn| {
        bn.iter(|| sort::argsort_f64(black_box(&fa), false))
    });
    g.bench_function("i64_std_sort_by_key", |bn| {
        bn.iter(|| {
            let v = ka.values();
            let mut idx: Vec<u32> = (0..m as u32).collect();
            idx.sort_by_key(|i| v[*i as usize]);
            idx
        })
    });
    g.finish();

    let names: Vec<String> = (0..200_000).map(|k| format!("customer-{k:08}")).collect();
    let ua = Utf8Array::from_options(names.iter().map(|s| Some(s.as_str()))).ok();
    let mut g = c.benchmark_group("hash");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("i64", |bn| bn.iter(|| hash::hash_i64(black_box(&i))));
    if let Some(ua) = &ua {
        g.throughput(Throughput::Elements(200_000));
        g.bench_function("utf8_200k", |bn| bn.iter(|| hash::hash_utf8(black_box(ua))));
    }
    g.finish();

    let d = PrimitiveArray::from_options((0..n as i64).map(|x| Some(x % 7)));
    let mut g = c.benchmark_group("arithmetic_i64");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("sub", |bn| {
        bn.iter(|| arithmetic::sub_i64(black_box(&i), &d))
    });
    g.bench_function("div_checked", |bn| {
        bn.iter(|| arithmetic::div_i64(black_box(&i), &d))
    });
    g.finish();

    if let Some(ua) = &ua {
        let mut g = c.benchmark_group("utf8_cmp_200k");
        g.throughput(Throughput::Elements(200_000));
        g.bench_function("scalar_gt", |bn| {
            bn.iter(|| {
                comparison::cmp_utf8_scalar(
                    black_box(ua),
                    comparison::CmpOp::Gt,
                    "customer-00100000",
                )
            })
        });
        g.finish();
    }

    if let Some(ua) = &ua {
        let mut g = c.benchmark_group("strings_200k");
        g.throughput(Throughput::Elements(200_000));
        g.bench_function("contains", |bn| {
            bn.iter(|| strings::contains(black_box(ua), "r-0001"))
        });
        g.bench_function("char_len", |bn| {
            bn.iter(|| strings::char_len(black_box(ua)))
        });
        g.bench_function("upper", |bn| bn.iter(|| strings::upper(black_box(ua))));
        g.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    targets = bench
}
criterion_main!(benches);
