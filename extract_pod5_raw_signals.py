"""Stream original ADC current from POD5 into a DataFrame-indexed signal store.

The original signal comes only from POD5. To avoid duplicating large arrays in
DataFrame object cells, signals are concatenated into one int16 binary file and
the CSV DataFrame stores each read's value offset and length. Conversion to pA:

    signal_pa = (signal_adc + calibration_offset) * calibration_scale

Example
-------
python extract_pod5_raw_signals.py --max-reads 10000

Load one read without copying the entire signal file:

    index = pandas.read_csv("read_signal_index.csv.gz")
    adc = numpy.memmap("raw_signal_adc.int16.bin", dtype="<i2", mode="r")
    row = index.iloc[0]
    signal = adc[row.signal_offset_values:row.signal_offset_values+row.signal_length]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pod5", type=Path,
        default=Path(r"/mnt/g/rna-mod-validation-all5mer-2026.07/subset_20m/Inosine/subset.pod5"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inosine_raw_signal_store"))
    parser.add_argument("--max-reads", type=int, default=None)
    args = parser.parse_args()
    if not args.pod5.exists():
        parser.error(f"POD5 does not exist: {args.pod5}")
    try:
        import pod5
    except ImportError as exc:
        raise SystemExit("Install the POD5 Python API: python -m pip install pod5") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.output_dir / "raw_signal_adc.int16.bin"
    index_path = args.output_dir / "read_signal_index.csv.gz"
    manifest_path = args.output_dir / "manifest.json"
    columns = [
        "read_id", "signal_offset_values", "signal_length", "sample_rate_hz",
        "calibration_offset", "calibration_scale", "pod5_source",
    ]
    count = 0
    value_offset = 0
    with pod5.Reader(args.pod5) as reader, signal_path.open("wb") as signal_handle, \
            gzip.open(index_path, "wt", newline="", encoding="utf-8") as index_handle:
        writer = csv.DictWriter(index_handle, fieldnames=columns)
        writer.writeheader()
        for read in reader.reads():
            adc = np.asarray(read.signal, dtype="<i2")
            signal_handle.write(adc.tobytes(order="C"))
            writer.writerow({
                "read_id": str(read.read_id),
                "signal_offset_values": value_offset,
                "signal_length": len(adc),
                "sample_rate_hz": read.run_info.sample_rate,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "pod5_source": str(args.pod5),
            })
            value_offset += len(adc)
            count += 1
            if args.max_reads is not None and count >= args.max_reads:
                break
    manifest = {
        "pod5": str(args.pod5.resolve()),
        "read_count": count,
        "signal_values": value_offset,
        "signal_dtype": "little-endian int16 ADC",
        "signal_file": str(signal_path.resolve()),
        "index_file": str(index_path.resolve()),
        "pa_formula": "(adc + calibration_offset) * calibration_scale",
        "complete_pod5": args.max_reads is None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

