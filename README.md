# Spectral optimization on Numerai financial time series

This repository records the complete research sequence testing whether a parameter-space spectral
gradient filter improves noisy financial prediction. It contains the original autonomous study,
the corrected parameter-covariance implementation, high-rank and overparameterized experiments,
and the final Numerai-comparable AdamW-versus-spectral study.

## Current conclusion

The latest evidence favours AdamW. A short historical outer holdout initially favoured the best
spectral candidate by about 0.003 mean CORR, but this did not generalise to Numerai's much larger,
later v5.3 diagnostic interval:

| Numerai v5.3 diagnostic, 642 eras | AdamW (`eden_adam`) | Spectral (`eden_eve`) |
|---|---:|---:|
| Mean CORR | **0.03606** | 0.01469 |
| CORR Sharpe | **1.9028** | 1.0702 |
| Mean BMC (informational, not a payout score) | **0.00203** | -0.00762 |
| Maximum feature exposure | **0.1915** | 0.6187 |

Both fixed Model Upload callables are deployed unstaked and submit automatically. The comparison
is not a clean optimizer-only causal test: the selected candidates differ in architecture,
regularisation, schedule, batch size, and repeated training exposure. In particular, the stopping
protocol extrapolated Eve from a barely winning 40.96M-example boundary checkpoint to a 107.85M
final refit, while Adam used 6.74M sampled examples. Both saw the same unique 574 training eras;
Eve repeatedly sampled them far more often.

The authoritative deployment record, hashes, diagnostics, caveats, and monitoring guidance are in
[`experiments/exp-008/live-artifacts/corrected-live-outcome.md`](experiments/exp-008/live-artifacts/corrected-live-outcome.md).

## Research sequence

| Stage | Purpose | Status / primary record |
|---|---|---|
| Experiments 001--005 | Original inter-sample $B \times B$ filtering study and audits | Historical; see [`paper/paper.pdf`](paper/paper.pdf) and [`audit/results-audit.md`](audit/results-audit.md) |
| Experiment 006 | Corrected parameter-space $p \times p$ filter and mechanism diagnostics | Complete; [`experiments/exp-006/results.md`](experiments/exp-006/results.md) |
| Experiment 007 | Overparameterized MLP, long training, ranks through the tractable high-rank range, and remove-direction ablations | Complete; [`experiments/exp-007/results.md`](experiments/exp-007/results.md) |
| Experiment 008 | Leakage-controlled Numerai HPO, AdamW comparison, production refits, upload, and diagnostics | Complete; [`experiments/exp-008/README.md`](experiments/exp-008/README.md) |

The original paper answers the original $B \times B$ experiment. It must not be read as a report
of the later $p \times p$ implementation or the final live candidates.

## Repository map

- [`experiments/`](experiments/) — executable code, frozen protocols, compact results, and plots.
- [`output/2026-08-10-numerai-competitive-spectral-submission/`](output/2026-08-10-numerai-competitive-spectral-submission/) — archived autonomous-research planning and evidence for Experiment 008; `PENDING` labels there record the pre-execution plan rather than current status.
- [`paper/`](paper/) — original-study LaTeX paper and compiled PDF.
- [`audit/`](audit/) — independent checks and rerun evidence for the original study.
- [`literature/`](literature/) — original literature review.
- [`challenge/`](challenge/) — adversarial reviews and limitation analysis.
- [`state.md`](state.md) — append-only autonomous workflow history.
- [`archive/researcher-state-recovery/`](archive/researcher-state-recovery/) — superseded failure/recovery snapshots retained for auditability, not current status.

Large datasets, model checkpoints, virtual environments, and generated submission bundles are
deliberately excluded from Git. Durable artifact identities needed for scientific provenance are
recorded as SHA-256 hashes in the experiment reports.

## Reproducing the current code checks

Experiment 008 is the maintained Python package:

```bash
cd experiments/exp-008
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
```

No repository command uploads a model or alters an NMR stake. The two completed uploads were
separate, explicitly authorized operations; staking remains disabled.
