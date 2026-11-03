"""Stage 2: build trainable Inosine read-site data from BAM + raw ADC store.

This script joins:
  1) a Dorado aligned BAM containing mv, ts and ns tags;
  2) read_signal_index.csv.gz and raw_signal_adc.int16.bin from POD5;
  3) inosine_candidate_sites.pkl created from FASTA/BED/ViennaRNA.

Outputs are sharded to avoid a huge DataFrame:
  metadata/part-*.parquet  - one row per read/site example
  tensors/signal-*.npy     - float32 [N, window_bases, points_per_base]
  tensors/mask-*.npy       - uint8   [N, window_bases]
  tensors/dwell-*.npy      - int32   [N, window_bases]

The default is a safe 100,000-primary-read pilot. Pass --max-reads 0 only after
checking the pilot. The full 20M-read input can generate hundreds of millions
of read-site rows, so negatives are balanced per read by default.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def platform_data_root() -> Path:
    linux = Path("/mnt/g/rna-mod-validation-all5mer-2026.07")
    return linux if linux.exists() else Path(r"G:\rna-mod-validation-all5mer-2026.07")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_signal_sqlite(csv_gz: Path, sqlite_path: Path, batch_size: int = 100_000) -> None:
    """Build a restart-safe SQLite lookup without loading 20M rows into RAM."""
    if sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as connection:
            try:
                ready = connection.execute(
                    "SELECT value FROM metadata WHERE key='complete'"
                ).fetchone()
                if ready and ready[0] == "1":
                    return
            except sqlite3.Error:
                pass
        raise RuntimeError(
            f"Incomplete SQLite index exists: {sqlite_path}. Remove only that file and rerun."
        )

    temporary = sqlite_path.with_suffix(sqlite_path.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE signals (
          read_id TEXT NOT NULL,
          signal_offset_values INTEGER NOT NULL,
          signal_length INTEGER NOT NULL,
          sample_rate_hz REAL NOT NULL,
          calibration_offset REAL NOT NULL,
          calibration_scale REAL NOT NULL
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    insert_sql = "INSERT INTO signals VALUES (?,?,?,?,?,?)"
    buffer: List[tuple] = []
    count = 0
    started = time.time()
    with gzip.open(csv_gz, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "read_id", "signal_offset_values", "signal_length", "sample_rate_hz",
            "calibration_offset", "calibration_scale",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"signal index missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            buffer.append((
                row["read_id"], int(row["signal_offset_values"]), int(row["signal_length"]),
                float(row["sample_rate_hz"]), float(row["calibration_offset"]),
                float(row["calibration_scale"]),
            ))
            if len(buffer) >= batch_size:
                connection.executemany(insert_sql, buffer)
                connection.commit()
                count += len(buffer)
                buffer.clear()
                print(f"indexed {count:,} signals ({time.time()-started:.1f}s)", flush=True)
        if buffer:
            connection.executemany(insert_sql, buffer)
            count += len(buffer)
            buffer.clear()
            connection.commit()
    print("creating unique read_id index...", flush=True)
    connection.execute("CREATE UNIQUE INDEX signals_read_id ON signals(read_id)")
    connection.execute("INSERT INTO metadata VALUES ('row_count', ?)", (str(count),))
    connection.execute("INSERT INTO metadata VALUES ('complete', '1')")
    connection.commit()
    connection.close()
    os.replace(temporary, sqlite_path)


def load_sites(path: Path) -> Tuple[pd.DataFrame, Dict[str, List[int]], Dict[Tuple[str, int], dict]]:
    frame = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path)
    required = {"site_id", "reference_name", "position_0based", "groundtruth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"site table missing columns: {sorted(required - set(frame.columns))}")
    by_reference: Dict[str, List[int]] = {}
    lookup: Dict[Tuple[str, int], dict] = {}
    for row in frame.to_dict("records"):
        reference = str(row["reference_name"])
        position = int(row["position_0based"])
        by_reference.setdefault(reference, []).append(position)
        lookup[(reference, position)] = row
    for positions in by_reference.values():
        positions.sort()
    return frame, by_reference, lookup


def split_for_reference(reference: str) -> str:
    """Group all reads/sites from one synthetic template into one split."""
    bucket = int.from_bytes(hashlib.blake2b(reference.encode(), digest_size=2).digest(), "little") % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def deterministic_negative_key(read_id: str, site_id: str) -> bytes:
    return hashlib.blake2b(f"{read_id}|{site_id}".encode(), digest_size=8).digest()


def choose_sites(
    rows: List[dict], read_id: str, negative_ratio: float, all_negatives: bool
) -> List[dict]:
    positives = [row for row in rows if int(row["groundtruth"]) == 1]
    negatives = [row for row in rows if int(row["groundtruth"]) == 0]
    if all_negatives:
        return positives + negatives
    # Reads without a covered positive site are not useful for balanced pilot training.
    if not positives:
        return []
    keep = min(len(negatives), int(math.ceil(len(positives) * negative_ratio)))
    negatives.sort(key=lambda row: deterministic_negative_key(read_id, str(row["site_id"])))
    return positives + negatives[:keep]


def decode_move_values(encoded: Sequence[int]) -> List[int]:
    """Decode Dorado's signed-int8 overflow continuation representation."""
    decoded: List[int] = []
    accumulator = 0
    continuing = False
    for raw in encoded:
        value = int(raw)
        accumulator += value
        if value in (-128, 127):
            continuing = True
            continue
        decoded.append(accumulator)
        accumulator = 0
        continuing = False
    if continuing:
        raise ValueError("truncated overflow-encoded move table")
    return decoded


def move_intervals(mv: Sequence[int], ts: int, ns: int, query_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Dorado mv tag into raw-read sample intervals in signal direction.

    Dorado defines the mapped interval as signal[ts:ns].
    """
    if len(mv) < 2:
        raise ValueError("empty mv tag")
    stride = int(mv[0])
    if stride <= 0 or ts < 0 or ns <= 0:
        raise ValueError("invalid stride/ts/ns")
    emission_blocks: List[int] = []
    decoded_moves = decode_move_values(mv[1:])
    for block_index, move_count in enumerate(decoded_moves):
        if int(move_count) < 0:
            raise ValueError("negative move count")
        emission_blocks.extend([block_index] * int(move_count))
    if len(emission_blocks) != query_length:
        raise ValueError(f"move emissions {len(emission_blocks)} != query length {query_length}")
    starts = ts + np.asarray(emission_blocks, dtype=np.int64) * stride
    if ns <= ts:
        raise ValueError("ns must be greater than ts")
    signal_end = ns
    ends = np.concatenate([starts[1:], np.asarray([signal_end], dtype=np.int64)])
    ends = np.maximum(ends, starts)
    return starts, ends


def resample_event(values: np.ndarray, points: int) -> Optional[np.ndarray]:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return np.full(points, float(values[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, points, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def reference_to_query_map(read) -> Dict[int, int]:
    """Map aligned reference positions to BAM query positions; deletions are absent."""
    mapping: Dict[int, int] = {}
    for query_position, reference_position in read.get_aligned_pairs(matches_only=False):
        if query_position is not None and reference_position is not None:
            mapping[int(reference_position)] = int(query_position)
    return mapping


def signal_base_index(query_position: int, query_length: int, is_reverse: bool, direct_rna: bool) -> int:
    """Map BAM query orientation to Dorado move-table signal orientation.

    DNA needs reversal for reverse alignments. Direct RNA additionally has its
    3'->5' signal direction reversed when sequence is written 5'->3'.
    """
    reverse = bool(is_reverse) ^ bool(direct_rna)
    return query_length - 1 - query_position if reverse else query_position


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        shard_rows: int,
        window_bases: int,
        points: int,
        metadata_format: str,
    ):
        self.output_dir = output_dir
        self.metadata_dir = output_dir / "metadata"
        self.tensor_dir = output_dir / "tensors"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = shard_rows
        self.window_bases = window_bases
        self.points = points
        self.metadata_format = metadata_format
        self.rows: List[dict] = []
        self.signals: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.dwells: List[np.ndarray] = []
        self.shard_index = 0
        self.total_rows = 0

    def append(self, metadata: dict, signal: np.ndarray, mask: np.ndarray, dwell: np.ndarray) -> None:
        metadata["tensor_shard"] = self.shard_index
        metadata["tensor_row"] = len(self.rows)
        self.rows.append(metadata)
        self.signals.append(signal)
        self.masks.append(mask)
        self.dwells.append(dwell)
        if len(self.rows) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        stem = f"part-{self.shard_index:05d}"
        final_signal = self.tensor_dir / f"signal-{stem}.npy"
        final_mask = self.tensor_dir / f"mask-{stem}.npy"
        final_dwell = self.tensor_dir / f"dwell-{stem}.npy"
        temp_signal = self.tensor_dir / f"signal-{stem}.npy.tmp"
        temp_mask = self.tensor_dir / f"mask-{stem}.npy.tmp"
        temp_dwell = self.tensor_dir / f"dwell-{stem}.npy.tmp"
        metadata = pd.DataFrame(self.rows)
        temp_metadata: Optional[Path] = None
        final_metadata: Optional[Path] = None
        try:
            with temp_signal.open("wb") as handle:
                np.save(handle, np.stack(self.signals).astype(np.float32))
            with temp_mask.open("wb") as handle:
                np.save(handle, np.stack(self.masks).astype(np.uint8))
            with temp_dwell.open("wb") as handle:
                np.save(handle, np.stack(self.dwells).astype(np.int32))

            selected_format = self.metadata_format
            if selected_format == "auto":
                selected_format = "parquet"
            if selected_format == "parquet":
                temp_metadata = self.metadata_dir / f"{stem}.parquet.tmp"
                final_metadata = self.metadata_dir / f"{stem}.parquet"
                try:
                    metadata.to_parquet(temp_metadata, index=False)
                except Exception as exc:
                    if self.metadata_format != "auto":
                        raise
                    print(
                        f"Parquet unavailable ({type(exc).__name__}: {exc}); "
                        "falling back to CSV.GZ.",
                        file=sys.stderr,
                        flush=True,
                    )
                    temp_metadata.unlink(missing_ok=True)
                    temp_metadata = self.metadata_dir / f"{stem}.csv.gz.tmp"
                    final_metadata = self.metadata_dir / f"{stem}.csv.gz"
                    metadata.to_csv(temp_metadata, index=False, compression="gzip")
            else:
                temp_metadata = self.metadata_dir / f"{stem}.csv.gz.tmp"
                final_metadata = self.metadata_dir / f"{stem}.csv.gz"
                metadata.to_csv(temp_metadata, index=False, compression="gzip")

            os.replace(temp_signal, final_signal)
            os.replace(temp_mask, final_mask)
            os.replace(temp_dwell, final_dwell)
            os.replace(temp_metadata, final_metadata)
        except Exception:
            for temporary in (temp_signal, temp_mask, temp_dwell, temp_metadata):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            raise
        self.total_rows += len(self.rows)
        print(f"wrote shard {self.shard_index}: {len(self.rows):,} rows", flush=True)
        self.shard_index += 1
        self.rows.clear(); self.signals.clear(); self.masks.clear(); self.dwells.clear()


def main() -> None:
    root = project_root()
    data_root = platform_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=Path, default=data_root / "basecalls_1/Inosine/Inosine.with_moves.bam")
    parser.add_argument("--signal-index", type=Path,
                        default=root / "outputs/inosine_raw_signal_store/read_signal_index.csv.gz")
    parser.add_argument("--signal-bin", type=Path,
                        default=root / "outputs/inosine_raw_signal_store/raw_signal_adc.int16.bin")
    parser.add_argument("--site-table", type=Path, default=root / "outputs/inosine_candidate_sites.pkl")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/inosine_read_site_dataset")
    parser.add_argument("--sqlite-index", type=Path, default=None)
    parser.add_argument("--max-reads", type=int, default=100_000,
                        help="primary mapped reads; 0 means no limit")
    parser.add_argument("--mapq", type=int, default=20)
    parser.add_argument("--signal-radius", type=int, default=10,
                        help="reference bases on each side; 10 gives 21 bases")
    parser.add_argument("--points-per-base", type=int, default=16)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--all-negatives", action="store_true")
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument(
        "--metadata-format",
        choices=("csv", "parquet", "auto"),
        default="csv",
        help="csv avoids PyArrow; auto tries Parquet and falls back on engine errors",
    )
    parser.add_argument("--direct-rna", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for path in (args.bam, args.signal_index, args.signal_bin, args.site_table):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.signal_radius < 0 or args.points_per_base < 1 or args.negative_ratio < 0:
        parser.error("invalid window/resampling/negative-ratio argument")

    # Fail before the potentially long 20M-row SQLite build when BAM support is absent.
    try:
        import pysam
    except ImportError as exc:
        raise SystemExit("Install BAM support first: python -m pip install pysam") from exc

    manifest_path = args.signal_index.parent / "manifest.json"
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_bytes = int(source_manifest["signal_values"]) * 2
        if args.signal_bin.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"ADC store size mismatch: {args.signal_bin.stat().st_size} != {expected_bytes}"
            )

    sqlite_path = args.sqlite_index or (args.output_dir / "read_signal_index.sqlite")
    build_signal_sqlite(args.signal_index, sqlite_path)
    signal_db = sqlite3.connect(sqlite_path)
    signal_db.row_factory = sqlite3.Row
    signal_store = np.memmap(args.signal_bin, dtype="<i2", mode="r")
    _site_frame, sites_by_ref, site_lookup = load_sites(args.site_table)

    window_bases = args.signal_radius * 2 + 1
    writer = ShardWriter(
        args.output_dir,
        args.shard_rows,
        window_bases,
        args.points_per_base,
        args.metadata_format,
    )
    stats = {
        "bam_records": 0, "primary_mapped_reads": 0, "reads_missing_signal_index": 0,
        "reads_missing_moves": 0, "reads_bad_moves": 0, "candidate_sites_covered": 0,
        "examples_written": 0, "events_missing": 0,
    }
    saw_mv = False
    with pysam.AlignmentFile(str(args.bam), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            stats["bam_records"] += 1
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < args.mapq:
                continue
            stats["primary_mapped_reads"] += 1
            if args.max_reads and stats["primary_mapped_reads"] > args.max_reads:
                break
            if not read.has_tag("mv"):
                stats["reads_missing_moves"] += 1
                continue
            saw_mv = True
            reference = read.reference_name
            positions = sites_by_ref.get(reference)
            if not positions:
                continue
            left = bisect.bisect_left(positions, read.reference_start)
            right = bisect.bisect_left(positions, read.reference_end)
            if left == right:
                continue
            ref_to_query = reference_to_query_map(read)
            covered_rows = [
                site_lookup[(reference, pos)] for pos in positions[left:right] if pos in ref_to_query
            ]
            stats["candidate_sites_covered"] += len(covered_rows)
            chosen = choose_sites(covered_rows, read.query_name, args.negative_ratio, args.all_negatives)
            if not chosen:
                continue
            signal_record = signal_db.execute(
                "SELECT signal_offset_values,signal_length,sample_rate_hz,calibration_offset,"
                "calibration_scale FROM signals WHERE read_id=?", (read.query_name,)
            ).fetchone()
            if signal_record is None:
                stats["reads_missing_signal_index"] += 1
                continue
            read_offset = int(signal_record["signal_offset_values"])
            read_length = int(signal_record["signal_length"])
            query_length = read.query_length
            if query_length is None:
                stats["reads_bad_moves"] += 1
                continue
            ts = int(read.get_tag("ts")) if read.has_tag("ts") else 0
            ns = int(read.get_tag("ns")) if read.has_tag("ns") else read_length - ts
            try:
                starts, ends = move_intervals(read.get_tag("mv"), ts, ns, query_length)
            except (ValueError, TypeError):
                stats["reads_bad_moves"] += 1
                continue
            if starts.min(initial=0) < 0 or ends.max(initial=0) > read_length:
                stats["reads_bad_moves"] += 1
                continue
            calibration_offset = float(signal_record["calibration_offset"])
            calibration_scale = float(signal_record["calibration_scale"])
            sm = float(read.get_tag("sm")) if read.has_tag("sm") else np.nan
            sd = float(read.get_tag("sd")) if read.has_tag("sd") else np.nan
            if not np.isfinite(sd) or sd <= 0:
                lo, hi = ts, min(read_length, ns)
                pa_all = (signal_store[read_offset+lo:read_offset+hi].astype(np.float32)
                          + calibration_offset) * calibration_scale
                sm = float(np.median(pa_all))
                mad = float(np.median(np.abs(pa_all - sm)))
                sd = max(1.4826 * mad, 1e-6)

            for site in chosen:
                center_ref = int(site["position_0based"])
                matrix = np.zeros((window_bases, args.points_per_base), dtype=np.float32)
                mask = np.zeros(window_bases, dtype=np.uint8)
                dwell = np.zeros(window_bases, dtype=np.int32)
                sample_starts: List[int] = []
                sample_ends: List[int] = []
                for row_index, ref_pos in enumerate(
                    range(center_ref - args.signal_radius, center_ref + args.signal_radius + 1)
                ):
                    query_pos = ref_to_query.get(ref_pos)
                    if query_pos is None:
                        stats["events_missing"] += 1
                        continue
                    base_index = signal_base_index(
                        query_pos, query_length, read.is_reverse, args.direct_rna
                    )
                    if not 0 <= base_index < query_length:
                        stats["events_missing"] += 1
                        continue
                    event_start, event_end = int(starts[base_index]), int(ends[base_index])
                    if event_end <= event_start:
                        stats["events_missing"] += 1
                        continue
                    adc = signal_store[
                        read_offset + event_start: read_offset + event_end
                    ].astype(np.float32)
                    pa = (adc + calibration_offset) * calibration_scale
                    normalized = (pa - sm) / sd
                    sampled = resample_event(normalized, args.points_per_base)
                    if sampled is None:
                        stats["events_missing"] += 1
                        continue
                    matrix[row_index] = sampled
                    mask[row_index] = 1
                    dwell[row_index] = event_end - event_start
                    sample_starts.append(event_start)
                    sample_ends.append(event_end)
                center_valid = bool(mask[args.signal_radius])
                if not center_valid:
                    continue
                metadata = {
                    "site_id": site["site_id"], "read_id": read.query_name,
                    "reference_name": reference, "position_0based": center_ref,
                    "position_1based": center_ref + 1, "groundtruth": int(site["groundtruth"]),
                    "split": split_for_reference(reference), "mapq": int(read.mapping_quality),
                    "cigar": read.cigarstring, "is_reverse_alignment": bool(read.is_reverse),
                    "query_position": int(ref_to_query[center_ref]),
                    "query_base": read.query_sequence[ref_to_query[center_ref]] if read.query_sequence else None,
                    "valid_signal_bases": int(mask.sum()),
                    "signal_sample_start_in_read": min(sample_starts),
                    "signal_sample_end_in_read": max(sample_ends),
                    "sample_rate_hz": float(signal_record["sample_rate_hz"]),
                    "calibration_offset": calibration_offset, "calibration_scale": calibration_scale,
                    "normalization_shift_sm": sm, "normalization_scale_sd": sd,
                }
                writer.append(metadata, matrix, mask, dwell)
                stats["examples_written"] += 1
    writer.flush()
    signal_db.close()
    if not saw_mv:
        raise RuntimeError(
            "No mv tag found in processed primary reads. Use the BAM created with Dorado --emit-moves."
        )
    run_manifest = {
        "inputs": {"bam": str(args.bam), "signal_index": str(args.signal_index),
                   "signal_bin": str(args.signal_bin), "site_table": str(args.site_table)},
        "parameters": vars(args) | {"bam": str(args.bam), "signal_index": str(args.signal_index),
                                    "signal_bin": str(args.signal_bin), "site_table": str(args.site_table),
                                    "output_dir": str(args.output_dir),
                                    "sqlite_index": str(sqlite_path)},
        "tensor_shape_per_example": [window_bases, args.points_per_base],
        "stats": stats,
        "total_output_rows": writer.total_rows,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
