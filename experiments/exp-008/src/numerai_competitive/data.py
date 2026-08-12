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
VALIDATION_SHA256 = "62bb9a587ecdcb5f3095809de276da381a803b699e905a82b962d2e4d35295c0"
VALIDATION_BENCH_SHA256 = "e771f395d689cee948435656af64bf789ea67c4421a5bc7cd568f83e85faf7d2"
FEATURES_SHA256 = "27de6b598b0d479415dba8062d050fc190469776f9c905feea9ee3f2bdda3631"
TARGETS = (
    "target", "target_cyrusd_20", "target_ender_20", "target_teager2b_20", "target_ender_60",
)
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


def _feature_codes(raw: np.ndarray, group: int) -> np.ndarray:
    """Encode the official five feature bins without silently guessing scale."""
    if np.issubdtype(raw.dtype, np.integer):
        if raw.min() < 0 or raw.max() > 4:
            raise ValueError(f"row group {group}: integer feature bins outside 0..4")
        return raw.astype(np.uint8, copy=False)
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
    return codes.astype(np.uint8)


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
        X[offset:offset + count] = _feature_codes(raw, group)
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
        "eras": len(np.unique(era)),
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


def _validate_freeze_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    required = {"status", "protocol", "code_commit", "search_sha256", "selected",
                "validation_reveal_authorized"}
    if not required <= set(value):
        raise ValueError(f"freeze manifest missing {sorted(required - set(value))}")
    if value["status"] != "frozen" or value["validation_reveal_authorized"] is not True:
        raise ValueError("validation remains sealed until an explicitly authorized frozen manifest")
    if set(value["selected"]) != {"adamw", "spectral"}:
        raise ValueError("freeze manifest must select exactly AdamW and spectral procedures")
    for arm in ("adamw", "spectral"):
        if not {"config_id", "updates", "seeds"} <= set(value["selected"][arm]):
            raise ValueError(f"freeze selection for {arm} is incomplete")
    return value


def build_validation_shard(source: Path, destination: Path, freeze_manifest: Path,
                           feature_set: str) -> Path:
    """Build the official validation shard only after procedure freeze."""
    if feature_set != "all":
        raise ValueError("sealed validation must use all features to support either frozen arm")
    source, freeze_manifest = source.resolve(), freeze_manifest.resolve()
    freeze = _validate_freeze_manifest(freeze_manifest)
    validation = source / "validation.parquet"
    benchmark = source / "validation_benchmark_models.parquet"
    features_json = source / "features.json"
    if sha256(validation) != VALIDATION_SHA256 or sha256(benchmark) != VALIDATION_BENCH_SHA256:
        raise ValueError("validation or validation-benchmark hash mismatch")
    features = _load_features(features_json, feature_set)
    validation_pf = pq.ParquetFile(validation)
    if validation_pf.metadata.num_rows != 4_107_040:
        raise ValueError("unexpected validation row count")
    benchmark_frame = pd.read_parquet(benchmark, columns=["era", *BENCHMARKS])
    if benchmark_frame.index.has_duplicates:
        raise ValueError("validation benchmark IDs are not unique")

    destination.mkdir(parents=True, exist_ok=True)
    n = validation_pf.metadata.num_rows
    X = np.lib.format.open_memmap(destination / "X_u8.npy", mode="w+", dtype=np.uint8,
                                  shape=(n, len(features)))
    y = np.lib.format.open_memmap(destination / "targets_f32.npy", mode="w+", dtype=np.float32,
                                  shape=(n, len(TARGETS)))
    era = np.lib.format.open_memmap(destination / "era_i16.npy", mode="w+", dtype=np.int16,
                                    shape=(n,))
    bench = np.lib.format.open_memmap(destination / "benchmarks_f32.npy", mode="w+",
                                      dtype=np.float32, shape=(n, len(BENCHMARKS)))
    ids_hash, offset = hashlib.sha256(), 0
    for group in range(validation_pf.metadata.num_row_groups):
        table = validation_pf.read_row_group(group, columns=["id", "era", *features, *TARGETS])
        ids_table, count = table["id"], table.num_rows
        _id_digest(ids_table, ids_hash)
        ids = pd.Index(ids_table.to_pylist(), name="id")
        joined = benchmark_frame.reindex(ids)
        if joined[BENCHMARKS].isna().any().any():
            raise ValueError(f"row group {group}: validation benchmark ID coverage is incomplete")
        group_eras = np.array([int(value) for value in table["era"].to_pylist()], dtype=np.int16)
        if not np.array_equal(joined["era"].astype(int).to_numpy(), group_eras):
            raise ValueError(f"row group {group}: validation benchmark eras disagree")
        raw = np.column_stack([table[name].to_numpy(zero_copy_only=False) for name in features])
        X[offset:offset + count] = _feature_codes(raw, group)
        for column, name in enumerate(TARGETS):
            y[offset:offset + count, column] = np.asarray(
                table[name].to_numpy(zero_copy_only=False), dtype=np.float32
            )
        era[offset:offset + count] = group_eras
        for column, name in enumerate(BENCHMARKS):
            bench[offset:offset + count, column] = joined[name].to_numpy(np.float32)
        offset += count
        print(json.dumps({"row_group": group, "rows_written": offset}), flush=True)
    for array in (X, y, era, bench):
        array.flush()
    resolved = {
        name: {
            "rows": int(np.isfinite(y[:, i]).sum()),
            "eras": len(np.unique(era[np.isfinite(y[:, i])])),
        }
        for i, name in enumerate(TARGETS)
    }
    manifest = {
        "split": "validation", "data_version": "v5.3", "round": 1329,
        "feature_set": feature_set, "feature_names": features, "targets": list(TARGETS),
        "benchmarks": list(BENCHMARKS), "rows": n, "eras": len(np.unique(era)),
        "era_min": int(era.min()), "era_max": int(era.max()), "resolved": resolved,
        "id_sequence_sha256": ids_hash.hexdigest(),
        "freeze_manifest_sha256": sha256(freeze_manifest), "freeze": freeze,
        "sources": {"validation.parquet": VALIDATION_SHA256,
                    "validation_benchmark_models.parquet": VALIDATION_BENCH_SHA256,
                    "features.json": FEATURES_SHA256},
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
    def open(cls, root: Path) -> TrainShard:
        return cls._open(root, "train")

    @classmethod
    def _open(cls, root: Path, expected_split: str) -> TrainShard:
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("split") != expected_split or manifest.get("data_version") != "v5.3":
            raise ValueError(f"loader accepts a frozen v5.3 {expected_split} shard only")
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


@dataclass
class ValidationShard(TrainShard):
    """Frozen official validation arrays; intentionally separate from development loading."""

    @classmethod
    def open(cls, root: Path) -> ValidationShard:
        shard = cls._open(root, "validation")
        manifest = shard.manifest
        if "freeze_manifest_sha256" not in manifest:
            raise ValueError("validation shard lacks freeze provenance")
        return shard


@dataclass
class ProductionShard(TrainShard):
    """Resolved train+validation data, unlocked only after sealed evaluation."""

    @classmethod
    def open(cls, root: Path) -> ProductionShard:
        shard = cls._open(root, "production_train")
        required = {
            "freeze_manifest_sha256", "sealed_evaluation_sha256",
            "resolved_validation_rows", "training_data_sha256",
        }
        missing = required - set(shard.manifest)
        if missing:
            raise ValueError(f"production shard lacks provenance: {sorted(missing)}")
        return shard
