//! Property tests: every kernel vs a naive Option-based reference.
use proptest::prelude::*;
use tessera_arrow::{Array, DataType, Utf8Array};
use tessera_arrow::{Bitmap, BooleanArray, PrimitiveArray};
use tessera_compute::{
    aggregate, arithmetic, boolean, cast, comparison::*, filter, hash, sort, strings, take,
};

fn opt_i64() -> impl Strategy<Value = Vec<Option<i64>>> {
    prop::collection::vec(prop::option::weighted(0.8, -1000i64..1000), 0..300)
}

fn opt_bool() -> impl Strategy<Value = Vec<Option<bool>>> {
    prop::collection::vec(prop::option::weighted(0.8, any::<bool>()), 0..300)
}

proptest! {
    #[test]
    fn boolean_ops(x in opt_bool(), y in opt_bool(), off in 0usize..10) {
        let n = x.len().min(y.len());
        let off = off.min(n);
        let a = BooleanArray::from_options(x[..n].iter().copied()).slice(off, n - off).unwrap();
        let b = BooleanArray::from_options(y[..n].iter().copied()).slice(off, n - off).unwrap();
        let zip = |f: fn(bool, bool) -> bool| -> Vec<Option<bool>> {
            x[off..n].iter().zip(&y[off..n]).map(|(p, q)| p.zip(*q).map(|(p, q)| f(p, q))).collect()
        };
        prop_assert_eq!(boolean::and(&a, &b).unwrap().iter().collect::<Vec<_>>(), zip(|p, q| p && q));
        prop_assert_eq!(boolean::or(&a, &b).unwrap().iter().collect::<Vec<_>>(), zip(|p, q| p || q));
        prop_assert_eq!(
            boolean::not(&a).unwrap().iter().collect::<Vec<_>>(),
            x[off..n].iter().map(|p| p.map(|p| !p)).collect::<Vec<_>>()
        );
    }
    #[test]
    fn sum_min_max_i64(v in opt_i64(), off in 0usize..10) {
        let a = PrimitiveArray::from_options(v.clone());
        let off = off.min(v.len());
        let s = a.slice(off, v.len() - off).unwrap();
        let r: Vec<i64> = v[off..].iter().flatten().copied().collect();
        prop_assert_eq!(aggregate::sum_i64(&s), (!r.is_empty()).then(|| r.iter().sum()));
        prop_assert_eq!(aggregate::min_i64(&s), r.iter().min().copied());
        prop_assert_eq!(aggregate::max_i64(&s), r.iter().max().copied());
    }

    #[test]
    fn take_matches_reference(
        v in opt_i64(),
        idx in prop::collection::vec(prop::option::weighted(0.85, 0u32..400), 0..200),
        off in 0usize..10,
    ) {
        let off = off.min(v.len());
        let s = PrimitiveArray::from_options(v.clone()).slice(off, v.len() - off).unwrap();
        let r = &v[off..];
        let idx: Vec<Option<u32>> = idx.into_iter().map(|i| i.filter(|i| (*i as usize) < r.len())).collect();
        let got: Vec<_> = take::take_primitive(&s, &idx).unwrap().iter().collect();
        let want: Vec<_> = idx.iter().map(|i| i.and_then(|i| r[i as usize])).collect();
        prop_assert_eq!(got, want);
        let strs: Vec<Option<String>> = r.iter().map(|x| x.map(|x| x.to_string())).collect();
        let u = Utf8Array::from_options(strs.iter().map(|s| s.as_deref())).unwrap();
        let got: Vec<_> = take::take_utf8(&u, &idx).unwrap().iter().map(|s| s.map(str::to_owned)).collect();
        let want: Vec<_> = idx.iter().map(|i| i.and_then(|i| strs[i as usize].clone())).collect();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn filter_utf8_boolean(v in opt_bool(), m in prop::collection::vec(prop::option::of(any::<bool>()), 300)) {
        let m = &m[..v.len()];
        let mask = BooleanArray::from_options(m.iter().copied());
        let a = BooleanArray::from_options(v.iter().copied());
        let want: Vec<_> = v.iter().zip(m).filter(|(_, k)| **k == Some(true)).map(|(x, _)| *x).collect();
        prop_assert_eq!(filter::filter_boolean(&a, &mask).unwrap().iter().collect::<Vec<_>>(), want);
        let strs: Vec<Option<&str>> = v.iter().map(|x| x.map(|b| if b { "yes" } else { "n" })).collect();
        let u = Utf8Array::from_options(strs.iter().copied()).unwrap();
        let want: Vec<_> = strs.iter().zip(m).filter(|(_, k)| **k == Some(true)).map(|(x, _)| *x).collect();
        let got = filter::filter_utf8(&u, &mask).unwrap();
        prop_assert_eq!(got.iter().collect::<Vec<_>>(), want);
    }

    #[test]
    fn argsort_matches_stable_reference(v in opt_i64(), desc in any::<bool>(), off in 0usize..10) {
        let off = off.min(v.len());
        let r = &v[off..];
        let s = PrimitiveArray::from_options(v.clone()).slice(off, v.len() - off).unwrap();
        // Reference: stable sort on (is_null, key) with the direction applied to the key only.
        let mut want: Vec<u32> = (0..r.len() as u32).collect();
        want.sort_by(|a, b| match (r[*a as usize], r[*b as usize]) {
            (Some(x), Some(y)) => if desc { y.cmp(&x) } else { x.cmp(&y) },
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        prop_assert_eq!(sort::argsort_i64(&s, desc).unwrap(), want.clone());
        let narrow = PrimitiveArray::from_options(r.iter().map(|x| x.map(|x| x as i32)));
        prop_assert_eq!(sort::argsort_i32(&narrow, desc).unwrap(), want.clone());
        let fl = PrimitiveArray::from_options(r.iter().map(|x| x.map(|x| x as f64 * 0.5)));
        prop_assert_eq!(sort::argsort_f64(&fl, desc).unwrap(), want.clone());
        let strs: Vec<Option<String>> = r.iter().map(|x| x.map(|x| format!("{:04}", x + 1000))).collect();
        let u = Utf8Array::from_options(strs.iter().map(|s| s.as_deref())).unwrap();
        prop_assert_eq!(sort::argsort_utf8(&u, desc).unwrap(), want);
    }

    #[test]
    fn argsort_i64_wide_keys(v in prop::collection::vec(any::<i64>(), 0..200)) {
        let a = PrimitiveArray::from_options(v.iter().map(|x| Some(*x)));
        let mut want: Vec<u32> = (0..v.len() as u32).collect();
        want.sort_by_key(|i| v[*i as usize]);
        prop_assert_eq!(sort::argsort_i64(&a, false).unwrap(), want);
    }

    #[test]
    fn hash_consistency(v in opt_i64(), off in 0usize..10) {
        let off = off.min(v.len());
        let r = &v[off..];
        let s = PrimitiveArray::from_options(v.clone()).slice(off, v.len() - off).unwrap();
        let h = hash::hash_i64(&s);
        prop_assert_eq!(h.len(), r.len());
        for i in 0..r.len() {
            match r[i] {
                None => prop_assert_eq!(h[i], hash::NULL_HASH),
                Some(x) => {
                    // equal values hash equal, and i32 agrees with i64
                    let one = PrimitiveArray::from_options([Some(x)]);
                    prop_assert_eq!(h[i], hash::hash_i64(&one)[0]);
                    let narrow = PrimitiveArray::from_options([Some(x as i32)]);
                    prop_assert_eq!(h[i], hash::hash_i32(&narrow)[0]);
                }
            }
        }
        let strs: Vec<Option<String>> = r.iter().map(|x| x.map(|x| x.to_string())).collect();
        let u = Array::Utf8(Utf8Array::from_options(strs.iter().map(|s| s.as_deref())).unwrap());
        let n = Array::Int64(s);
        let both = hash::hash_columns(&[&n, &u]).unwrap();
        // Rows with equal (a, b) pairs get equal hashes (deterministic).
        for i in 0..r.len() {
            for j in 0..i.min(8) {
                if r[i] == r[j] {
                    prop_assert_eq!(both[i], both[j]);
                }
            }
        }
    }

    #[test]
    fn cast_matches_reference(v in opt_i64(), off in 0usize..10) {
        let off = off.min(v.len());
        let r = &v[off..];
        let a = Array::Int64(PrimitiveArray::from_options(v.clone())).slice(off, v.len() - off).unwrap();
        let Array::Int32(n) = cast::cast(&a, DataType::Int32).unwrap() else { panic!("type") };
        prop_assert_eq!(n.iter().collect::<Vec<_>>(), r.iter().copied().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>());
        let Array::Float64(f) = cast::cast(&a, DataType::Float64).unwrap() else { panic!("type") };
        prop_assert_eq!(f.iter().collect::<Vec<_>>(), r.iter().map(|x| x.map(|x| x as f64)).collect::<Vec<_>>());
        // i64 -> utf8 -> i64 and i64 -> f64 -> i64 round-trip exactly in this range
        let s = cast::cast(&a, DataType::Utf8).unwrap();
        let Array::Int64(want) = a else { panic!("type") };
        let Array::Int64(rt) = cast::cast(&s, DataType::Int64).unwrap() else { panic!("type") };
        prop_assert_eq!(rt.iter().collect::<Vec<_>>(), want.iter().collect::<Vec<_>>());
        let Array::Int64(back) = cast::cast(&Array::Float64(f), DataType::Int64).unwrap() else { panic!("type") };
        prop_assert_eq!(back.iter().collect::<Vec<_>>(), want.iter().collect::<Vec<_>>());
    }

    #[test]
    fn string_kernels(
        v in prop::collection::vec(prop::option::weighted(0.8, "[aAbß\u{e9} ]{0,8}"), 0..100),
        pat in "[ab]{0,2}",
        off in 0usize..10,
    ) {
        let off = off.min(v.len());
        let r = &v[off..];
        let a = Utf8Array::from_options(v.iter().map(|s| s.as_deref())).unwrap().slice(off, v.len() - off).unwrap();
        prop_assert_eq!(
            strings::contains(&a, &pat).unwrap().iter().collect::<Vec<_>>(),
            r.iter().map(|s| s.as_ref().map(|s| s.contains(pat.as_str()))).collect::<Vec<_>>()
        );
        prop_assert_eq!(
            strings::char_len(&a).unwrap().iter().collect::<Vec<_>>(),
            r.iter().map(|s| s.as_ref().map(|s| s.chars().count() as i32)).collect::<Vec<_>>()
        );
        let up = strings::upper(&a).unwrap();
        prop_assert_eq!(
            up.iter().map(|s| s.map(str::to_owned)).collect::<Vec<_>>(),
            r.iter().map(|s| s.as_ref().map(|s| s.to_uppercase())).collect::<Vec<_>>()
        );
    }

    #[test]
    fn sub_div_and_utf8_cmp(
        x in opt_i64(),
        y in opt_i64(),
        words in prop::collection::vec(prop::option::weighted(0.8, "[ab]{0,3}"), 0..100),
        k in "[ab]{0,3}",
    ) {
        let n = x.len().min(y.len());
        let a = PrimitiveArray::from_options(x[..n].iter().copied());
        let b = PrimitiveArray::from_options(y[..n].iter().copied());
        let pair = |f: &dyn Fn(i64, i64) -> Option<i64>| -> Vec<Option<i64>> {
            x[..n].iter().zip(&y[..n]).map(|(p, q)| p.zip(*q).and_then(|(p, q)| f(p, q))).collect()
        };
        prop_assert_eq!(arithmetic::sub_i64(&a, &b).unwrap().iter().collect::<Vec<_>>(), pair(&|p, q| Some(p - q)));
        prop_assert_eq!(
            arithmetic::div_i64(&a, &b).unwrap().iter().collect::<Vec<_>>(),
            pair(&|p, q| (q != 0).then(|| p / q))
        );
        let u = Utf8Array::from_options(words.iter().map(|w| w.as_deref())).unwrap();
        let got = cmp_utf8_scalar(&u, CmpOp::LtEq, &k).unwrap();
        prop_assert_eq!(
            got.iter().collect::<Vec<_>>(),
            words.iter().map(|w| w.as_ref().map(|w| w.as_str() <= k.as_str())).collect::<Vec<_>>()
        );
        let both = cmp_utf8(&u, CmpOp::Gt, &u).unwrap();
        prop_assert_eq!(
            both.iter().collect::<Vec<_>>(),
            words.iter().map(|w| w.as_ref().map(|_| false)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn f64_paths_agree(v in prop::collection::vec(-1000i32..1000, 0..500)) {
        // Integer-valued floats make every summation order exact.
        let f: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        let s = aggregate::sum_f64_scalar(&f);
        prop_assert_eq!(aggregate::sum_f64_chunked(&f), s);
        prop_assert_eq!(aggregate::sum_f64_avx2(&f), s);
        let a = PrimitiveArray::from_options(f.iter().map(|x| Some(*x)));
        let mn = f.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.min(x))));
        prop_assert_eq!(aggregate::min_f64(&a), mn);
    }

    #[test]
    fn filter_matches_reference(v in opt_i64(), m in prop::collection::vec(prop::option::of(any::<bool>()), 300)) {
        let a = PrimitiveArray::from_options(v.clone());
        let mask = BooleanArray::from_options(m[..v.len()].iter().copied());
        let got: Vec<_> = filter::filter_primitive(&a, &mask).unwrap().iter().collect();
        let want: Vec<_> = v.iter().zip(&m).filter(|(_, k)| **k == Some(true)).map(|(x, _)| *x).collect();
        prop_assert_eq!(got, want);
        let bits: Vec<u64> = v.iter().map(|x| x.unwrap_or(0) as u64).collect();
        let sel = Bitmap::from_bools(m[..v.len()].iter().map(|k| *k == Some(true)));
        prop_assert_eq!(filter::filter_u64_avx2(&bits, &sel), filter::filter_slice_scalar(&bits, &sel));
    }

    #[test]
    fn cmp_and_add(v in opt_i64(), k in -1000i64..1000) {
        let a = PrimitiveArray::from_options(v.clone());
        let got: Vec<_> = cmp_scalar(&a, CmpOp::LtEq, k).iter().collect();
        prop_assert_eq!(got, v.iter().map(|x| x.map(|x| x <= k)).collect::<Vec<_>>());
        let sum: Vec<_> = arithmetic::add_i64(&a, &a).unwrap().iter().collect();
        prop_assert_eq!(sum, v.iter().map(|x| x.map(|x| x * 2)).collect::<Vec<_>>());
    }
}
