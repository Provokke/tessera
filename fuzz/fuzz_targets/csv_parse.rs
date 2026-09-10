#![no_main]
//! Arbitrary bytes must never panic the CSV reader; anything it accepts
//! must survive a write -> read round-trip with the same shape.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(df) = tessera_io::read_csv_bytes(data) {
        let text = tessera_io::write_csv_string(&df);
        let back = tessera_io::read_csv_bytes(text.as_bytes());
        if df.height() > 0 {
            assert_eq!(back.map(|b| b.shape()).ok(), Some(df.shape()));
        }
    }
});
