# Exp-008: Numerai-comparable AdamW versus spectral optimization

This directory is the executable, no-leakage pipeline for the v5.3 Numerai Tournament study.
The primary target is explicitly the v5.3 main `target`; the primary endpoint is exact standalone
Numerai CORR. Historical validation is not itself a resolved current live leaderboard reputation.
No command here uploads a model, submits predictions, or stakes NMR.

## Reproduce integrity checks

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
```

The base immutable search is `configs/search-v1.json`; its promotion rules and the dated,
pre-validation high-rank amendment are recorded in `fidelity-protocol.md`. The amendment creates
hashed source-selection, GPU-probe and augmented-search artifacts under `results/`; only the
augmented search may reach final freeze. Development runners accept only v5.3 train shards. The official
validation builder requires a provenance-verified procedure/model freeze manifest and is
unavailable to HPO.

## Frozen HPO sequence

For each outer fold, run the exact stages below. Selection requires equal split/seed coverage
and deterministically ranks mean CORR, worst-cell CORR, lower config ID, then lower budget.

1. Both arms, all 40 paired config IDs, earliest inner fold, seed 0, 5,000 updates.
2. Both arms over the top-12 paired union, every eligible inner fold, seed 0, 20,000 updates.
3. Both arms over the top-4 paired union, every eligible inner fold and seeds 0/1/2 at
   5,000/20,000/100,000 updates, plus GPU-audited high-rank spectral variants at the 100,000
   budget needed to activate them. Exact earlier cells are reused, not rerun.
4. One configuration/update-budget winner per arm, refit on the outer-training block at its
   selected budget and score the untouched outer block at seeds 0/1/2.

On MATS, synchronize source only between stages, then build the shared frozen environment once:

```bash
sbatch slurm/sync-env.sbatch
bash slurm/launch-outer.sh outer_2 ENV_JOB_ID
bash slurm/submit-selected.sh SELECTION.json 20000 0 ENV_JOB_ID SPLIT...
```

`launch-outer.sh` creates the exact 80-cell paired F0 manifest, starts its audited monitor, and
starts the fold-specific promotion waiter. It refuses an existing manifest, summary, or controller
session rather than duplicating a procedure.
`promote-f0-when-ready.sh` waits for the audited F0 table before submitting F1.
`promote-f1-when-ready.sh` likewise requires every audited F1 temporal-fold table, selects the
top-four paired union, selects the best development architecture feasible at rank 2,048,
GPU-probes ranks 512/1,024/1,536/2,048/4,096, and submits the exact asymmetric three-budget F2
manifest. Caught CUDA OOM rejects only that probed rank; any other probe error stops promotion.
All three promotion scripts accept an optional `outer_1`, `outer_2`, or `outer_3` argument and
derive the required two, three, or four eligible inner folds. Fold-specific monitor, supervisor,
selection, manifest, and audit names prevent one outer procedure from satisfying another's gate.

Submission scripts print tab-separated job provenance. `monitor-stage.sh` refuses incomplete
coverage and dependency failures. GPU jobs use the dependency-built environment with `--no-sync`
so concurrent jobs never mutate it. Jobs receive `SIGUSR1` one hour before the free-partition
walltime and atomically checkpoint; `resume-checkpointed-stage.sh` resubmits only those exact
manifested tasks without changing their config, seed, fold, or update target. The budgeted F2
stage runs under `supervise-resumable-stage.sh`, which repeats exact checkpoint resumptions until
every manifested result exists and only then creates the audited fold/seed summaries.
The selected nested-outer estimate is a different operation: `submit-outer-eval.sh` runs only
the independently selected config for each arm on the named untouched outer split. Its manifest
uses the same resumable supervisor with `--skip-summary` and must pass the dedicated outer-result
audit before entering `oof.py`. `submit-refit.sh` is reserved for the later all-train deployment
refit and must never be used as nested-outer evidence.
`promote-f2-when-ready.sh` applies that route automatically only after every audited F2
budget/fold/seed summary exists; `audit-outer-when-ready.sh` then blocks until all six arm-specific
outer results exist and runs the dedicated audit.

For an unattended complete nested run, start `slurm/continue-nested-pipeline.sh` in tmux while
outer 1 is active. It waits for each audited outer result, submits a fresh environment gate before
launching outer 2 and outer 3, builds the audited OOF/candidate handoff, and starts final canonical
selection/refits. It stops at the final-refit audit: validation is still revealed only through the
separate immutable-freeze command.

After all outer folds, aggregate only untouched nested-outer predictions:

```bash
bash slurm/build-oof-candidate.sh
```

The script requires all three completed outer audits, verifies each selection against its audit,
resolves exactly nine seed results per arm, and then writes the nested-outer estimate and frozen
train-only candidate plan. It refuses to overwrite either output.

The candidate selector tests predeclared model weights 0.1/0.25/0.5/0.75/1.0 against the
main-target-matched Ender60 benchmark.
An arm is eligible only if its standalone signal is positive in every nested outer fold. The
selected transformation is frozen before validation; it cannot alter the primary standalone
spectral-minus-AdamW claim.

The final train-only selection starts only after all three outer audits exist:

```bash
bash slurm/launch-final-selection.sh DEPENDENCY_JOB_ID
```

It verifies each audited outer winner, forms their paired union, and evaluates both optimizers on
all four canonical train-era folds and all three seeds. Exact result cells already produced by a
nested stage are reused only by path; the exact-ID summarizer re-audits their frozen config,
signature, prediction schema, and hashes. Final summaries use a separate `summary-final-*`
namespace, so they cannot overwrite the outer-fold selection evidence.
Once every exact configuration/budget final-selection summary exists, the promotion waiter selects
one configuration/update pair per arm, submits seeds 0/1/2 for 60-day-purged validation refit,
resumes only exact
checkpointed refits, and writes `audit-final-refits-budgeted/refit-audit.json`. The audit checks selection/manifest coverage,
frozen-search identity, all-train era provenance, signatures, model metadata, state-dict shapes,
and parameter counts before the later freeze computes full model-file hashes.

## One-time validation and live bundle

After final canonical-fold selection and three 60-day-purged validation refits per arm, create
the freeze:

```bash
uv run python -m numerai_competitive.code_snapshot create \
  --repo ../.. --source-prefix experiments/exp-008 --commit COMMIT \
  --output code-snapshot.json
uv run python -m numerai_competitive.freeze \
  --search results/search-v1-high-rank.json --protocol fidelity-protocol.md \
  --candidate-plan out/candidate-plan.json --output out/freeze.json \
  --code-commit COMMIT --code-snapshot code-snapshot.json --code-root . \
  --adamw-config ID --spectral-config ID \
  --adamw-model MODEL ... --spectral-model MODEL ... \
  --adamw-updates ADAMW_BUDGET --spectral-updates SPECTRAL_BUDGET \
  --seed 0 --seed 1 --seed 2 --authorize-validation-reveal
```

The snapshot hashes the committed execution surface (`src`, `configs`, `slurm`, lockfile and
protocol) from Git itself. Freeze independently requires the rsynced cluster tree to have exactly
that file set and byte content; a supplied SHA is not accepted as provenance by itself.

Only then build the validation shard with `--feature-set all` and invoke
`numerai_competitive.evaluate_frozen`. The all-feature shard is required because independently
selected AdamW and spectral winners may use different feature subsets; each frozen model selects
its own named columns during inference. The evaluator emits
raw per-era predictions, rolling/cumulative plots, BMC, era-wise benchmark correlations, and
moving-block intervals for spectral-minus-AdamW and each candidate-minus-Ender comparison.

The operational sealed handoff is:

```bash
bash slurm/submit-sealed-evaluation.sh FULL_40_CHARACTER_CODE_COMMIT
```

It refuses to submit until the three-seed refit audit and frozen train-only candidate agree with
the final winners. The first Slurm job creates and hashes the immutable freeze, only then queries
Numerai for the exact pinned v5.3 validation, benchmark, and feature artifacts, verifies their
frozen hashes, and builds the all-feature validation shard. A dependent GPU job runs the one-time
evaluation and writes `official-validation/evaluation-complete.json`. Neither job contains upload,
submission, or staking code.

After sealed evaluation, refit the already-frozen winning procedure on all currently resolved
labels. This expands training data but does not reopen hyperparameter, arm, budget, blend or seed
selection:

```bash
uv run python -m numerai_competitive.code_snapshot create \
  --repo REPOSITORY_ROOT --source-prefix PATH/TO/experiments/exp-008 \
  --commit FULL_40_CHARACTER_PRODUCTION_COMMIT --output production-code-snapshot.json
bash slurm/submit-production-refits.sh \
  FULL_40_CHARACTER_PRODUCTION_COMMIT ENVIRONMENT_JOB_ID
```

The frozen manifest retains the earlier procedure commit used for search and sealed validation;
the production audit separately records and verifies the production implementation commit. This
avoids pretending that models trained before the live-refit addition used later source bytes.

Export the separately audited production seed ensemble and frozen blend as a Model Upload callable:

```bash
uv run python -m numerai_competitive.live \
  --model MODEL_SEED_0 --model MODEL_SEED_1 --model MODEL_SEED_2 \
  --freeze out/freeze.json --production-audit out/production-refit-audit.json \
  --output out/predictor.pkl
uv run python -m numerai_competitive.validate_live \
  --callable out/predictor.pkl --live live.parquet \
  --benchmark live_benchmark_models.parquet --output out/live-runtime.json
uv run python -m numerai_competitive.predict_live \
  --callable out/predictor.pkl --live live.parquet \
  --benchmark live_benchmark_models.parquet --output out/live_predictions.csv
```

The validator uses one-thread model inference, enforces the ten-minute and 4 GB artifact/RSS
limits, and checks the exact target-free row schema. The final CSV remains local until the user
separately authorizes upload or submission.
The exported pickle serializes the project-specific predictor and MLP implementation by value;
it does not require `numerai_competitive` to be installed in Numerai's execution image. A
subprocess test blocks every import of that package while loading the pickle. Before any upload,
the final artifact must additionally pass the official `numerai-predict` Python 3.12 container,
whose current interface accepts a one- or two-argument callable and whose documented default
contract is one CPU, 4 GB RAM, and ten minutes.

On the desktop, clone the official repository at a recorded commit and run its unmodified
Python 3.12 Dockerfile against the exact live fixture:

```bash
bash slurm/validate-official-container.sh NUMERAI_PREDICT_CHECKOUT \
  results/live-bundle/predictor.pkl results/live-bundle/live-fixture/live.parquet \
  results/live-bundle/live-fixture/live_benchmark_models.parquet \
  results/live-bundle/live_predictions.csv results/live-bundle/official-container
```

The wrapper builds the content-addressed official source, disables container networking, limits
execution to one CPU, 4,000,000,000 bytes and 600 seconds, and checks the official runner's IDs
and predictions against the independently generated conventional CSV. The audit records the full
official Git commit, Docker image ID, hashes, runtime, and numerical agreement.

After sealed evaluation, build and resource-test the target-free candidate without uploading it:

```bash
bash slurm/submit-live-bundle.sh FULL_40_CHARACTER_CODE_COMMIT
```

The first dependent job exports the audited production refits of the exact frozen arm/seed
procedure and downloads `live.parquet`
and `live_benchmark_models.parquet` within one unchanged current round. It records both hashes and
checks unique aligned IDs, absence of targets, and the frozen benchmark column. The validation job
uses one CPU, a 4 GB allocation, the complete unprojected live DataFrame (matching the official
runner), and a ten-minute application limit. The same timed inference writes the conventional
`live_predictions.csv`; no upload occurs. The resulting pickle and fixture can then be copied to
the desktop for the mandatory official `numerai-predict` Python 3.12 Docker test.

After nested-outer aggregation and the sealed evaluation are complete, render the scale-safe
comparison report:

```bash
uv run python -m numerai_competitive.final_report \
  --outer results/nested-outer/nested-outer-report.json \
  --validation results/official-validation/official-validation-report.json \
  --leaderboard leaderboard/leaderboard-summary.json --freeze results/freeze.json \
  --search results/search-v1-high-rank.json \
  --admission results/base-search-memory-admission.json --output results/final-report
```

The report directly compares only models scored on the same historical target and eras. It shows
the exact frozen AdamW and spectral configurations, the 40-pair frozen base search, its
outcome-independent memory-admission count, and every audited high-rank amendment candidate.
It also shows
the dated public one-year live reputations in a separate context section and explicitly refuses to
infer a live rank from historical main-`target` validation CORR.

After the official-container audit has been copied into `results/live-bundle/official-container`,
cross-check the complete evidence chain:

```bash
uv run python -m numerai_competitive.completion_audit \
  --results results --leaderboard leaderboard/leaderboard-summary.json \
  --output results/completion-audit.json
```

This independently re-hashes the three nested outer audits, outer-winner union, final selection
and validation refits, immutable validation model files, freeze, sealed validation artifacts,
production-data/refit audit and production model files, target-free live fixture,
resource-tested callable and predictions, official Docker output, leaderboard snapshot, and final
report inputs. A green stage marker alone is insufficient for the final completion claim.
