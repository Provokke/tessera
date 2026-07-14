//! Property test: write_csv -> read_csv reproduces the DataFrame.
use proptest::prelude::*;
use tessera_arrow::{Array, PrimitiveArray, Utf8Array};
use tessera_core::{DataFrame, Series};
use tessera_io::{read_csv_bytes, write_csv_string};

proptest! {
    #[test]
    fn roundtrip(rows in prop::collection::vec(
        (prop::option::of(any::<i64>()), prop::option::of(-1e12f64..1e12), prop::option::of("[a-z][a-z ,\"\n]{0,6}|")),
        1..60)) {
        // Guarantee each column has a non-null value so inference is unambiguous.
        let mut rows = rows;
        rows[0] = (Some(rows[0].0.unwrap_or(1)), Some(rows[0].1.unwrap_or(0.5)), Some("seed".into()));
        let df = DataFrame::new(vec![
            Series::new("i", Array::Int64(PrimitiveArray::from_options(rows.iter().map(|r| r.0)))),
            Series::new("f", Array::Float64(PrimitiveArray::from_options(rows.iter().map(|r| r.1)))),
            Series::new("s", Array::Utf8(Utf8Array::from_options(rows.iter().map(|r| r.2.as_deref())).unwrap())),
        ]).unwrap();
        let text = write_csv_string(&df);
        let back = read_csv_bytes(text.as_bytes()).unwrap();
        prop_assert_eq!(back.shape(), df.shape());
        for (a, b) in df.columns().iter().zip(back.columns()) {
            prop_assert_eq!(a.data_type(), b.data_type());
            for r in 0..df.height() {
                prop_assert_eq!(a.get(r), b.get(r));
            }
        }
    }
}

proptest! {
    #[test]
    fn parallel_equals_sequential(lines in prop::collection::vec((prop::option::of(any::<i32>()), "[a-z]{0,5}", prop::option::of(-1e6f64..1e6)), 0..80),
                                  crlf in any::<bool>(), chunks in 2usize..9) {
        let nl = if crlf { "\r\n" } else { "\n" };
        let mut text = format!("a,b,c{nl}");
        for (a, b, c) in &lines {
            let a = a.map(|x| x.to_string()).unwrap_or_default();
            let c = c.map(|x| format!("{x:?}")).unwrap_or_default();
            text.push_str(&format!("{a},{b},{c}{nl}"));
        }
        let seq = tessera_io::read_csv_bytes_chunked(text.as_bytes(), 1);
        let par = tessera_io::read_csv_bytes_chunked(text.as_bytes(), chunks);
        prop_assert_eq!(seq, par);
    }
}

fn gen_text(lines: &[(Option<i32>, String, Option<f64>)]) -> String {
    let mut text = String::from("a,b,c\n");
    for (a, b, c) in lines {
        let a = a.map(|x| x.to_string()).unwrap_or_default();
        let c = c.map(|x| format!("{x:?}")).unwrap_or_default();
        text.push_str(&format!("{a},{b},{c}\n"));
    }
    text
}

proptest! {
    /// Projection == reading everything then selecting, in the requested order,
    /// sequential and parallel.
    #[test]
    fn projection_equals_select(lines in prop::collection::vec((prop::option::of(any::<i32>()), "[a-z]{0,5}", prop::option::of(-1e6f64..1e6)), 0..80),
                                pick in prop::sample::subsequence(vec!["a", "b", "c"], 1..=3),
                                rev in any::<bool>(), chunks in 1usize..9) {
        let text = gen_text(&lines);
        let mut pick = pick;
        if rev { pick.reverse(); }
        let full = tessera_io::read_csv_bytes_chunked(text.as_bytes(), 1).unwrap();
        let want = full.select(&pick).unwrap();
        let opts = tessera_io::CsvOptions::new().with_projection(pick.clone()).with_chunks(chunks);
        let got = tessera_io::read_csv_bytes_with(text.as_bytes(), &opts).unwrap();
        prop_assert_eq!(want, got);
    }

    /// Inference over the first N rows == inference on a file holding only those
    /// N rows; later rows either parse as that type or give `ParseField`.
    #[test]
    fn infer_first_n_rows(lines in prop::collection::vec((prop::option::of(any::<i32>()), "[a-z]{0,5}", prop::option::of(-1e6f64..1e6)), 1..60),
                          n in 0usize..70, chunks in 1usize..9) {
        let text = gen_text(&lines);
        let prefix = gen_text(&lines[..n.min(lines.len())]);
        let pre = tessera_io::read_csv_bytes_chunked(prefix.as_bytes(), 1).unwrap();
        let opts = tessera_io::CsvOptions::new().with_infer_rows(n).with_chunks(chunks);
        let seq = tessera_io::read_csv_bytes_with(text.as_bytes(), &opts.clone().with_chunks(1));
        let par = tessera_io::read_csv_bytes_with(text.as_bytes(), &opts);
        prop_assert_eq!(&seq, &par);
        match seq {
            Ok(df) => {
                for (a, b) in pre.columns().iter().zip(df.columns()) {
                    prop_assert_eq!(a.data_type(), b.data_type());
                }
                prop_assert_eq!(df.height(), lines.len());
                for (a, b) in pre.columns().iter().zip(df.columns()) {
                    for r in 0..pre.height() {
                        prop_assert_eq!(a.get(r), b.get(r));
                    }
                }
            }
            Err(e) => {
                let is_parse_field = matches!(e, tessera_io::CsvError::ParseField { .. });
                prop_assert!(is_parse_field);
                // Only possible when rows beyond the window exist.
                prop_assert!(n < lines.len());
            }
        }
    }
}

#[test]
fn read_csv_from_path_and_errors() {
    let dir = std::env::temp_dir().join(format!("tessera_io_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("t.csv");
    std::fs::write(&p, "x,y\n1,a\n2,b\n").unwrap();
    let df =
        tessera_io::read_csv(&p, &tessera_io::CsvOptions::new().with_projection(["y"])).unwrap();
    assert_eq!(df.shape(), (2, 1));
    assert!(matches!(
        tessera_io::read_csv(dir.join("missing.csv"), &tessera_io::CsvOptions::new()),
        Err(tessera_io::CsvError::Io(_))
    ));
    assert!(matches!(
        tessera_io::read_csv_bytes_with(
            b"x\n1\n",
            &tessera_io::CsvOptions::new().with_projection(["nope"])
        ),
        Err(tessera_io::CsvError::UnknownColumn(_))
    ));
    // Row 3 is outside the 2-row inference window and not an integer.
    let e = tessera_io::read_csv_bytes_with(
        b"x\n1\n2\nz\n",
        &tessera_io::CsvOptions::new().with_infer_rows(2),
    )
    .unwrap_err();
    assert!(
        matches!(e, tessera_io::CsvError::ParseField { record: 3, .. }),
        "{e:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
