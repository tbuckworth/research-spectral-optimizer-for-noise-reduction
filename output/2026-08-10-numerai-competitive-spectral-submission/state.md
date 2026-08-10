---
run_id: 2026-08-10-numerai-competitive-spectral-submission
topic: Build and honestly evaluate a competitive Numerai submission, optimizing AdamW first and then testing whether spectral optimization improves it
current_step: 2
status: literature_search_in_progress
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
---

# Workflow Progress

## Step 1: Clarification

Scope is frozen from the user's explicit requested sequence and the prior project context. The work is autonomous within free compute and reversible repository changes. A live unstaked upload will be prepared, but any actual external submission will be surfaced before execution.

## Step 2: Literature and Platform Contract

The search plan was generated and approved autonomously under the user's instruction to proceed. Three disjoint searches covered official Numerai sources, methodological/academic sources, and high-quality community evidence. Official sources fix the current dataset at v5.3, live CORR20v2/MMC at the unreleased `target_cyrus_20`, and official benchmark walk-forward geometry at 156-era prediction blocks with an 8-era purge for 20D targets. The downloaded round-1329 v5.3 generic `target` instead exactly aliases `target_ender_60`; released historical Diagnostics targets and current live payout targets must therefore be reported separately. The existing local NumPy shard has no provenance metadata and is excluded from final comparable claims.
