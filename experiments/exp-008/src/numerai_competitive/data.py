"""Sealed v5.3 train shard construction and memory-mapped loading."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRAIN_SHA256 = "bae773ddd7eea6ed07c55d87b882cc061901abbb724972b2a630381417c328f8"
TRAIN_BENCH_SHA256 = "5a8729941c481abe95236920663e1d3cb1140407cc743e3a7f71afbf78f80248"
FEATURES_SHA256 = "27de6b598b0d479415dba8062d050fc190469776f9c905feea9ee3f2bdda3631"
TARGETS = ("target_cyrusd_20", "target_ender_20", "target_teager2b_20", "target_ender_60")
BENCHMARKS = ("v53_lgbm_ender20", "v53_lgbm_ender60")


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_features(path: Path, feature_set: str) -> list[str]:
    if sha256(path) != FEATURES_SHA256:
        raise ValueError("features.json hash does not match frozen round-1329 manifest")
    metadata = json.loads(path.read_text())
    return list(metadata["feature_sets"][feature_set])


def _id_digest(values, digest: hashlib._Hash) -> None:
    for value in values.to_pylist():
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")


def build_train_shard(source: Path, destination: Path, feature_set: str = "all") -> Path:
    """Convert frozen train Parquet to compact quantized memmaps.

    This function refuses any non-train filenames and verifies source hashes before
    reading targets. It has no validation-data parameter by design.
    """
    source = source.resolve()
    train = source / "train.parquet"
    benchmark = source / "train_benchmark_models.parquet"
    features_json = source / "features.json"
    if train.name != "train.parquet" or "validation" in str(train).lower():
        raise ValueError("development shard accepts frozen train.parquet only")
    if sha256(train) != TRAIN_SHA256 or sha256(benchmark) != TRAIN_BENCH_SHA256:
        raise ValueError("train or train-benchmark hash mismatch")
    features = _load_features(features_json, feature_set)
    train_pf = pq.ParquetFile(train)
    if train_pf.metadata.num_rows != 2_746_268:
        raise ValueError("unexpected train row count")
    benchmark_frame = pd.read_parquet(benchmark, columns=["era", *BENCHMARKS])
    if benchmark_frame.index.has_duplicates:
        raise ValueError("train benchmark IDs are not unique")

    destination.mkdir(parents=True, exist_ok=True)
    n = train_pf.metadata.num_rows
    X = np.lib.format.open_memmap(destination / "X_u8.npy", mode="w+", dtype=np.uint8,
                                  shape=(n, len(features)))
    y = np.lib.format.open_memmap(destination / "targets_f32.npy", mode="w+", dtype=np.float32,
                                  shape=(n, len(TARGETS)))
    era = np.lib.format.open_memmap(destination / "era_i16.npy", mode="w+", dtype=np.int16,
                                    shape=(n,))
    bench = np.lib.format.open_memmap(destination / "benchmarks_f32.npy", mode="w+",
                                      dtype=np.float32, shape=(n, len(BENCHMARKS)))
    ids_hash = hashlib.sha256()
    offset = 0
    for group in range(train_pf.metadata.num_row_groups):
        train_table = train_pf.read_row_group(group, columns=["id", "era", *features, *TARGETS])
        train_ids = train_table["id"]
        _id_digest(train_ids, ids_hash)
        count = train_table.num_rows
        ids = pd.Index(train_ids.to_pylist(), name="id")
        joined_benchmark = benchmark_frame.reindex(ids)
        train_eras = np.array([int(value) for value in train_table["era"].to_pylist()],
                              dtype=np.int16)
        # The round-1329 artifact starts at era 0158 (one era later than the
        # illustrative 0157 boundary in the current benchmark documentation).
        expected_coverage = train_eras >= 158
        actual_coverage = joined_benchmark[BENCHMARKS[0]].notna().to_numpy()
        if not np.array_equal(actual_coverage, expected_coverage):
            raise ValueError(f"row group {group}: benchmark ID coverage does not match eras >=158")
        joined_eras = joined_benchmark["era"].dropna().astype(int).to_numpy()
        if not np.array_equal(joined_eras, train_eras[actual_coverage]):
            raise ValueError(f"row group {group}: benchmark eras disagree after ID join")
        raw = np.column_stack([
            train_table[name].to_numpy(zero_copy_only=False) for name in features
        ])
        # Current v5.3 Parquet stores int8 bin codes 0..4. Retain support for
        # float quarter-bin files, but never guess between the two silently.
        if np.issubdtype(raw.dtype, np.integer):
            if raw.min() < 0 or raw.max() > 4:
                raise ValueError(f"row group {group}: integer feature bins outside 0..4")
            codes = raw.astype(np.uint8, copy=False)
        else:
            raw = np.nan_to_num(raw, nan=(2.0 if np.nanmax(raw) > 1 else 0.5))
            if raw.min() >= 0 and raw.max() <= 1:
                codes = np.rint(raw * 4.0)
                reconstructed = codes / 4.0
            else:
                codes = np.rint(raw)
                reconstructed = codes
            if (codes.min() < 0 or codes.max() > 4 or
                    not np.allclose(raw, reconstructed, atol=1e-6)):
                raise ValueError(f"row group {group}: features are not expected quarter bins")
        X[offset:offset + count] = codes.astype(np.uint8)
        for column, name in enumerate(TARGETS):
            y[offset:offset + count, column] = np.asarray(
                train_table[name].to_numpy(zero_copy_only=False), dtype=np.float32
            )
        era[offset:offset + count] = train_eras
        for column, name in enumerate(BENCHMARKS):
            bench[offset:offset + count, column] = joined_benchmark[name].to_numpy(np.float32)
        offset += count
        print(json.dumps({"row_group": group, "rows_written": offset}), flush=True)
    for array in (X, y, era, bench):
        array.flush()
    manifest = {
        "split": "train",
        "data_version": "v5.3",
        "round": 1329,
        "feature_set": feature_set,
        "feature_names": features,
        "targets": list(TARGETS),
        "benchmarks": list(BENCHMARKS),
        "rows": n,
        "eras": int(len(np.unique(era))),
        "era_min": int(era.min()),
        "era_max": int(era.max()),
        "id_sequence_sha256": ids_hash.hexdigest(),
        "sources": {
            "train.parquet": TRAIN_SHA256,
            "train_benchmark_models.parquet": TRAIN_BENCH_SHA256,
            "features.json": FEATURES_SHA256,
        },
    }
    atomic_json(destination / "manifest.json", manifest)
    return destination / "manifest.json"


@dataclass
class TrainShard:
    root: Path
    X: np.ndarray
    targets: np.ndarray
    eras: np.ndarray
    benchmarks: np.ndarray
    manifest: dict

    @classmethod
    def open(cls, root: Path) -> "TrainShard":
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("split") != "train" or manifest.get("data_version") != "v5.3":
            raise ValueError("development loader accepts a frozen v5.3 train shard only")
        X = np.load(root / "X_u8.npy", mmap_mode="r")
        targets = np.load(root / "targets_f32.npy", mmap_mode="r")
        eras = np.load(root / "era_i16.npy", mmap_mode="r")
        benchmarks = np.load(root / "benchmarks_f32.npy", mmap_mode="r")
        n = manifest["rows"]
        if any(len(value) != n for value in (X, targets, eras, benchmarks)):
            raise ValueError("shard row counts disagree with manifest")
        return cls(root, X, targets, eras, benchmarks, manifest)

    def target_index(self, name: str) -> int:
        return self.manifest["targets"].index(name)

    def benchmark_index(self, name: str) -> int:
        return self.manifest["benchmarks"].index(name)

    def row_indices(self, era_values: tuple[str, ...] | list[str]) -> np.ndarray:
        wanted = np.array([int(x) for x in era_values], dtype=self.eras.dtype)
        return np.flatnonzero(np.isin(self.eras, wanted))
