---
run_id: 2026-08-10-numerai-competitive-spectral-submission
topic: Build and honestly evaluate a competitive Numerai submission, optimizing AdamW first and then testing whether spectral optimization improves it
current_step: 9
status: experiments_in_progress
agent_backend: codex
agent_model: gpt-5.6-sol
clarifications:
  - q: Which leaderboard and product are in scope?
    a: The flagship Numerai Tournament, not Numerai Signals or Numerai Crypto.
  - q: What does comparable performance mean?
    a: Use the current official Numerai dataset, payout target, scoring implementation, validation-era protocol, and official benchmark-model predictions; distinguish offline diagnostics from genuinely live leaderboard reputation.
  - q: What does success look like?
    a: First produce the strongest honestly tuned AdamW baseline, then determine whether spectral optimization beats it under the same search and evaluation budget, and prepare an unstaked live-submission candidate that is plausibly competitive.
  - q: What prior context should be retained?
    a: The prior study established that spectral filtering suppresses overfitting in a 7.49M-parameter MLP, but AdamW had almost no hyperparameter optimization and the historical test metric was not directly leaderboard-comparable.
  - q: What are the compute and operational constraints?
    a: Use the desktop RTX 3090 and free MATS compute by default. Paid elastic GPUs require separate approval. Do not stake NMR or upload a live model without explicit authorization; preparing submission-ready artifacts is in scope.
decisions:
  - step: 1
    decision: Use strict temporal model selection with no final-holdout reuse; tune AdamW before exposing spectral to the final comparison.
  - step: 1
    decision: Treat official benchmark models and Numerai Diagnostics-compatible scores as the offline comparator; treat the live leaderboard as requiring forward submissions over time.
  - step: 3
    decision: Novelty is PARTIALLY_NOVEL; frame this as a controlled Numerai domain-transfer study of an existing streaming gradient-covariance eigenspace filter.
  - step: 4
    decision: Use separate offline Diagnostics-compatible and prospective live-leaderboard success gates; never equate released historical targets with the unreleased live payout target.
  - step: 6
    decision: Accept MAJOR_REVISIONS — symmetric complete-procedure searches, nested outer development evaluation, one primary target/endpoint, no runtime fairness, and no asymmetric spectral efficacy stop.
  - step: 8
    decision: Autonomous fail-fast agreement confirmed; execute Exp-008A through Exp-008E on free compute, with official validation sealed until freeze.
---

# Workflow Progress

## Step 1: Clarification

Scope is frozen from the user's explicit requested sequence and the prior project context. The work is autonomous within free compute and reversible repository changes. A live unstaked upload will be prepared, but any actual external submission will be surfaced before execution.

## Step 2: Literature and Platform Contract

The search plan was generated and approved autonomously under the user's instruction to proceed. Three disjoint searches covered official Numerai sources, methodological/academic sources, and high-quality community evidence. Official sources fix the current dataset at v5.3, live CORR20v2/MMC at the unreleased `target_cyrus_20`, and official benchmark walk-forward geometry at 156-era prediction blocks with an 8-era purge for 20D targets. The downloaded round-1329 v5.3 generic `target` instead exactly aliases `target_ender_60`; released historical Diagnostics targets and current live payout targets must therefore be reported separately. The existing local NumPy shard has no provenance metadata and is excluded from final comparable claims.

## Step 3: Novelty

Verdict: PARTIALLY_NOVEL. The optimizer already exists and has mixed evidence on synthetic/image tasks. The contribution is a mechanism-accurate, equal-budget, leakage-resistant domain-transfer study on the current Numerai panel, not a claim to have invented a spectral optimizer.

## Step 4: Success Criteria

Success criteria are frozen in `success-criteria.md`. They require exact official metrics, named targets, a sealed official validation set, a real AdamW HPO, an equal-budget spectral comparison, benchmark-relative diagnostics, reproducibility, and prospective live evidence before any leaderboard claim.

## Steps 5–8: Decomposition, Challenge, Plan, Agreement

The fail-fast decomposition was independently challenged by assumption, mentor and pre-mortem reviews. The initial plan received MAJOR_REVISIONS because it partly matched wall-clock cost, gave spectral an asymmetric efficacy screen, lacked nested evaluation of the selection procedure, and exposed too many primary routes. All fix-now items are incorporated in the governing challenge revisions and `planned-experiments.md`. The autonomous execution agreement is confirmed under the user's instruction to plan and execute.

## Step 9: Experiments in progress

F0 completed all 80 paired cells on `outer_1_inner_1` at 5,000 updates and seed 0. A
post-run audit initially halted because the jobs predated the embedded `search_config_id`
schema. No result was accepted by directory name alone: all 80 were subsequently verified
against the frozen search's full scientific configuration, the official feature-set dimensions,
their split metadata and recomputed signatures, and all prediction artifacts were schema-checked
and hashed. Spectral won 18/40 pairs but had mean paired CORR delta −0.002653; its best cell was
0.050616 versus the best AdamW cell at 0.048384. These are selection-fold results only.

The independently ranked top-12 lists formed a 16-configuration paired union. The exact 64-cell
F1 manifest (16 configurations × 2 arms × 2 inner folds, 20,000 updates, seed 0) passed its
environment dependency and ran on the free MATS `compute` partition. As of
2026-08-11T11:20:18Z, 61/64 result cells were complete. The remaining three were spectral cells
30, 35 and 39 on `outer_1_inner_2`; Slurm accounting showed advancing CPU time and stable memory,
so they were active rather than stalled. An independent waiter requires both exact 32-cell inner
summaries before forming the paired top-four union and submitting F2. Official validation remains
absent and sealed.

A deterministic final report generator is now tested. It will compare nested-outer and sealed
official-validation scores only on the common historical target/scorer, and place the dated public
live leaderboard snapshot in a visibly separate context section. It refuses to translate
historical `target_cyrusd_20` CORR into a live rank; direct rank comparability still requires
resolved prospective unstaked rounds.

The current official `numerai/numerai-predict` Python 3.12 image was also built locally from full
commit `2b6eb43612db25bcf047608b4c1fd61cc3ff3c06`. Its content-addressed Docker image ID and exact
runtime versions are frozen in `official-runner-snapshot.json`. The image build alone is not a
compatibility result: the final candidate must still execute on the complete live fixture under
the one-CPU, 4,000,000,000-byte and 600-second limits, and its output must match the independently
generated prediction path.
