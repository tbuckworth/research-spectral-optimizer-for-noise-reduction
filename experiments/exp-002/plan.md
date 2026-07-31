# exp-002: Sequence-arm vmap feasibility (hand-rolled GRU cell)

**Component**: #2 (P=0.35, lambda=1.05).
**Fail semantics**: NOT fail-fast. A FAIL here is a pre-authorized scope
decision (drop to MLP-only minimum viable contribution); the run always
continues to exp-003 regardless of this outcome.

## Question

Can per-sample gradients be computed through a recurrent/sequence model with
torch.func vmap(grad(...)), correctly and fast enough, so a second-architecture
arm (sequence model) is feasible? torch.func vmap does NOT compose with
cuDNN-backed nn.LSTM/nn.GRU — a functional, hand-rolled cell is required.
This is the least-tested code path in the plan.

## Task (local CPU — this is a correctness smoke test, not a GPU experiment)

1. Implement a ~1-layer GRU cell as pure functions (functional form usable
   under torch.func: `functional_call` over a params dict, or explicit
   parameter-passing functions). Sequence-to-one regression head (predict a
   scalar target from a short sequence).
2. Build synthetic Numerai-shaped within-era sequence data (e.g. sequences of
   length ~10-20 of ~50-300-dim feature vectors, continuous target). Real
   Numerai data is NOT required — this tests the code path, not the science.
3. Compute per-sample gradients with vmap(grad(per_sample_loss)) over a batch
   (B ≥ 64) and verify they match a plain per-sample python-loop reference to
   ~1e-5 (max abs difference, all parameter tensors).
4. Time it on CPU; project the cost of a full sequence-model training run
   (subsampled data, B ≤ 1024 per-sample grads + B×B eigh per step) on one
   L40, using the CPU-vs-L40 scaling observed for the MLP path if exp-001's
   timing data is available (`../exp-001/results.md`), else a conservative
   documented estimate.
5. Optional (only if the projection is borderline): one 5-min --qos=debug GPU
   job on the cluster to measure throughput directly, following the compute
   profile's staging workflow.

Integration target: the same SpectralConsensusFilter
(`/home/titus/pyg/optimizers/experiments/spectral_optimizer.py`, read-only —
copy what you need into this directory) must be able to consume these
per-sample gradients; demonstrate one filtered step end-to-end on the GRU
model (CPU is fine).

## Pass / Fail

PASS: vmap(grad) per-sample gradients through the functional GRU cell match
the per-sample loop reference to ~1e-5 AND projected sequence-model training
run <25 min on one L40, AND one end-to-end filtered optimizer step runs.

FAIL: vmap errors requiring >2 h of surgery, or projected runtime over the
job cap. Do not sink more than ~2 h total into surgery — a FAIL is cheap and
pre-authorized.

## Amendment F13 (binding, for downstream claim wording)

If the eventual sequence arm would run on OHLCV rather than within-era
Numerai, the claim is "second setting", NOT "architecture consistency",
unless a cheap MLP-on-OHLCV arm is added. Note in results.md which claim the
feasibility result supports.

## Deliverables

- `results.md`: PASS/FAIL, max-abs-diff vs loop reference, CPU timings,
  projected L40 runtime with the scaling assumption stated, end-to-end
  filtered-step confirmation, any vmap incompatibilities found and their
  workarounds.
- Code left in this directory. No files >50 MB.
