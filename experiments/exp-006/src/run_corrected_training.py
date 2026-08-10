#!/usr/bin/env python3
"""Run the existing frozen Numerai protocol with the corrected local filter."""
from __future__ import annotations

import json
import os
import time

import numpy as np

import driver
from spectral_filter_fixed import StableSpectralGradientFilter


RANKS = [1, 2, 4, 8, 16, 32]
RELATIVE_EIG_TOL = 1e-8
ORIGINAL_RANK_CONFIG = driver.rank_config


def rank_config(rank, replica, protocol):
    cfg = ORIGINAL_RANK_CONFIG(rank, replica, protocol)
    cfg["optimizer"] = "AdamW+StableSpectralGradientFilter"
    cfg["relative_eig_tol"] = RELATIVE_EIG_TOL
    cfg["stabilize_every"] = 100
    return cfg


def make_filter(model, optimizer, cfg):
    if cfg.get("arm") != "B":
        return None
    return StableSpectralGradientFilter(
        model, optimizer, rank=cfg["rank"], decay=cfg.get("decay", .99),
        warmup=cfg.get("warmup", 100), normalize="none", weighting="hard",
        alpha=1.0, soft_residual=True, adaptive="none",
        relative_eig_tol=cfg["relative_eig_tol"],
        stabilize_every=cfg["stabilize_every"])


def run_rank_sweep(protocol, X, Y, E, train, valid):
    monitor = driver.deterministic_valid_monitor(valid, E, protocol)
    records = []
    for rank in RANKS:
        for replica in driver.SWEEP_REPLICAS:
            cfg = rank_config(rank, replica, protocol)
            records.append(driver.train_trial(
                cfg, f"B-r{rank}-rep{replica}", train, valid, monitor,
                X, Y, E, protocol, 2000))
    assert len(records) == len(RANKS) * len(driver.SWEEP_REPLICAS)
    by_rank = {}
    for rank in RANKS:
        rr = [x for x in records if x["config"]["rank"] == rank]
        scores = [x["terminal_valid_mean_numerai_corr"] for x in rr]
        by_rank[str(rank)] = {
            "rank": rank, "n": len(rr), "mean_valid": float(np.mean(scores)),
            "std_valid": float(np.std(scores, ddof=1)), "scores": scores,
            "mean_seconds": float(np.mean([x["seconds"] for x in rr])),
            "mean_seconds_per_step": float(np.mean([x["seconds_per_step"] for x in rr])),
            "realized_k_median": float(np.mean([x["diagnostics"]["realized_k_median"] for x in rr])),
            "median_kept_norm": float(np.mean([x["diagnostics"]["median_kept_norm"] for x in rr])),
            "median_cosine": float(np.mean([x["diagnostics"]["median_cosine"] for x in rr]))}
    winner = max(RANKS, key=lambda rank: (by_rank[str(rank)]["mean_valid"], -rank))
    selected = rank_config(winner, 0, protocol); selected.pop("sweep_replica_seed")
    summary = {"selection_scope": "TRAIN-to-VALID only; TEST untouched",
        "fixed_intervention": "stable hard p×p filter; relative eig tol=1e-8",
        "ranks": RANKS, "replicas": driver.SWEEP_REPLICAS,
        "by_rank": by_rank, "selected_rank": winner,
        "selected_config": selected}
    driver.atomic_json(driver.OUT/"rank-sweep-summary.json", summary)
    print("RANK_SWEEP_SELECTED", json.dumps(summary), flush=True)
    return summary, selected


def main():
    driver.SpectralGradientFilter = StableSpectralGradientFilter
    driver.RANKS = RANKS
    driver.rank_config = rank_config
    driver.make_filter = make_filter
    driver.OUT.mkdir(exist_ok=True); driver.STATE.mkdir(exist_ok=True)
    start = time.time()
    protocol, sha = driver.protocol_and_guards()
    driver.runtime_checks(protocol)
    shard = driver.find_shard(protocol)
    X, Y, E = driver.open_shard(shard)
    eras = np.asarray(E)
    train = np.where(eras <= 791)[0]
    valid = np.where((eras >= 796) & (eras <= 891))[0]
    refit = np.where(eras <= 891)[0]
    test = np.where((eras >= 896) & (eras <= 1005))[0]
    adamw_selected, adamw_refits, adamw_tests = driver.load_existing_adamw()
    sweep, selected = run_rank_sweep(protocol, X, Y, E, train, valid)
    if os.environ.get("STOP_AFTER_SWEEP", "1") == "1":
        print("VALIDATION_GATE_STOP", json.dumps(sweep), flush=True)
        return
    comparison = driver.paired_comparison(
        protocol, X, Y, E, refit, test, selected, adamw_refits, adamw_tests)
    driver.cleanup_models()
    summary = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha,
        "unit": "EXP-006-STABLE", "fold": 1,
        "relative_eig_tol": RELATIVE_EIG_TOL,
        "existing_adamw_config": adamw_selected["config"],
        "rank_sweep": sweep, "comparison": comparison,
        "verdict": "COMPLETE", "wall_s": time.time()-start,
        "test_touches": len(driver.read_jsonl(driver.OUT/"test-touch.jsonl"))}
    driver.atomic_json(driver.OUT/"summary.json", summary)
    print("STABLE_FILTER_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__": main()
