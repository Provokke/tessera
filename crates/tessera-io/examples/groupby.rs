//! Read a small CSV and aggregate it: `cargo run -p tessera-io --example groupby`.
use tessera_core::AggKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = tessera_io::read_csv_bytes(b"k,v\na,1\nb,\na,3\n")?;
    let g = df.group_by(&["k"], &[("v", AggKind::Sum), ("v", AggKind::Count)])?;
    println!("{g}");
    Ok(())
}
