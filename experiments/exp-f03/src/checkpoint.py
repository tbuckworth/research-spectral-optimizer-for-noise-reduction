#!/usr/bin/env python3
"""Refit checkpoint save/restore for the fold jobs (B2 deliverable).

SpectralGradientFilter has no state_dict()/load_state_dict(), so these helpers
define the checkpointable filter state explicitly:

  V           (p, k) streaming top eigenvector basis (device tensor)
  S           (k,)   singular values (kept on CPU by the filter)
  proj_k      adaptive projection width (int or None)
  step_count  filter step counter (drives warmup gating)
  grad_mean   (p,)   EMA of the gradient (centering term, device tensor)

plus the constructor hyperparameters (for a restore-time consistency check),
the model + base-optimizer state_dicts, every RNG stream the training loop
consumes (torch global CPU, the per-run data-order torch.Generator, numpy
global, python random), and the loop step index.

Checkpoint writes are ATOMIC: serialize to <path>.tmp, fsync, os.replace, then
fsync the directory -- a kill mid-checkpoint leaves the previous checkpoint
intact, never a torn file.
"""
import os
import random

import numpy as np
import torch

FILTER_HPARAMS = ["rank", "decay", "warmup", "filter_strength",
                  "energy_threshold", "adaptive", "normalize", "weighting",
                  "alpha", "soft_residual"]


def filter_state_dict(filt):
    return {
        "V": None if filt.V is None else filt.V.detach().clone(),
        "S": None if filt.S is None else filt.S.detach().clone(),
        "proj_k": filt.proj_k,
        "step_count": filt.step_count,
        "grad_mean": (None if filt.grad_mean is None
                      else filt.grad_mean.detach().clone()),
        "hparams": {k: getattr(filt, k) for k in FILTER_HPARAMS},
    }


def load_filter_state_dict(filt, state):
    for k, v in state["hparams"].items():
        assert getattr(filt, k) == v, (
            f"filter hparam mismatch on restore: {k}={getattr(filt, k)!r} "
            f"vs checkpoint {v!r}")
    filt.V = None if state["V"] is None else state["V"].clone()
    filt.S = None if state["S"] is None else state["S"].clone()
    filt.proj_k = state["proj_k"]
    filt.step_count = state["step_count"]
    filt.grad_mean = (None if state["grad_mean"] is None
                      else state["grad_mean"].clone())


def rng_state_dict(data_generator=None):
    s = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if data_generator is not None:
        s["data_generator"] = data_generator.get_state()
    return s


def load_rng_state_dict(state, data_generator=None):
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if data_generator is not None:
        assert "data_generator" in state, "checkpoint lacks data_generator state"
        data_generator.set_state(state["data_generator"])


def save_checkpoint(path, *, model, optimizer, filt, step, data_generator=None,
                    extra=None):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "filter": filter_state_dict(filt),
        "rng": rng_state_dict(data_generator),
        "extra": extra or {},
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(ckpt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))            # atomic swap
    dfd = os.open(os.path.dirname(os.path.abspath(str(path))) or ".",
                  os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_checkpoint(path, *, model, optimizer, filt, data_generator=None):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    load_filter_state_dict(filt, ckpt["filter"])
    load_rng_state_dict(ckpt["rng"], data_generator)
    return ckpt["step"], ckpt["extra"]
