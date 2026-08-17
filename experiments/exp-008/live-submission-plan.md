# Unstaked Numerai live-candidate plan

Status: preparation, training, bundle export, and official-container validation are complete.
No upload, submission, staking, deletion, or replacement of a Numerai model is authorized by
this document. See `live-artifacts/readiness-audit.md` for the final evidence summary.

## Objective

Create two prospective, simultaneously started Numerai Tournament controls:

1. the frozen bounded-study AdamW winner (configuration 38, 20,000 updates); and
2. the frozen bounded-study spectral winner (configuration 39, 20,000 updates).

Refit each procedure with seeds 0, 1, and 2 on every currently resolved v5.3 main-target era,
export a CPU inference ensemble, verify it under Numerai's official Model Upload limits, and make
the two artifacts ready for an explicitly authorized unstaked upload. Keeping both arms provides
a contemporaneous control for live market regime.

## Frozen scientific choices

- Source selection: `results/selection-one-day-confirmed-top1.json` and the completed
  `outer_1` audit.
- AdamW: configuration 38, 20,000 updates, seeds 0/1/2.
- Spectral: configuration 39, 20,000 updates, seeds 0/1/2.
- Target: v5.3 main `target`; feature set: `all`.
- The outer result is immutable evidence and is not reused as a training score.
- Production refits may use all labels resolved at the refit cutoff because architecture,
  optimizer, rank, budget, blend, and seeds are already frozen.
- Initial live operation is unstaked. No early live score may change one arm independently.

## Execution stages and gates

### 1. Freeze and provenance

- Record the full Git commit and code-snapshot hashes.
- Materialize a production-data manifest with dataset names, hashes, target, feature order,
  resolved-era cutoff, and row count.
- Assert that no live/unresolved row enters training.

Gate: procedure selection, code snapshot, data manifest, and six-cell refit manifest agree.

### 2. GPU production refits

- Submit six time-limited jobs to the free MATS `compute` partition.
- Save models/checkpoints/results under durable NFS storage.
- Resume only exact checkpointed jobs after a walltime interruption.
- Do not use paid elastic partitions without separate approval.

Gate: all six results are complete, finite, match frozen config IDs/budgets/seeds, and have
audited model-file hashes and state-dict shapes.

### 3. CPU live bundles

- Export one three-seed ensemble callable per optimizer.
- The callable accepts Numerai live features (and benchmark models when supplied) and returns one
  aligned `prediction` column in [0, 1].
- Evaluate seeds sequentially and batch rows to remain comfortably below 4 GB RAM.

Gate: deterministic predictions, exact IDs, no missing/non-finite values, and no target access.

### 4. Official runtime validation

- Test each exact pickle in Numerai's official Python 3.12 `numerai-predict` container.
- Disable network access; constrain to one CPU, 4,000,000,000 bytes, and 600 seconds.
- Compare container output numerically with independently generated conventional CSV output.

Gate: both bundles pass schema, memory, runtime, import-isolation, and numerical-equivalence
checks. A failed bundle is fixed and re-audited; it is never uploaded speculatively.

### 5. Human account setup and explicit authorization

The user creates/signs into a Numerai account, creates two fresh Tournament model slots, and
creates a least-privilege API key with `View user info`, `View historical submission info`, and
`Upload submissions and pickled models`. Credentials are stored outside Git.

Confirmed slot names:

- `eden_eve` (spectral)
- `eden_adam` (AdamW)

Gate: user confirms the exact slot IDs and explicitly authorizes uploading the two audited
pickles. Staking remains disabled.

### 6. Upload and prospective monitoring

- Upload both audited Model Upload callables in the same round.
- Verify Numerai execution logs and accepted submissions for both slots.
- Record round, artifact hashes, slot IDs, upload time, and server status.
- Monitor CORR20v2, BMC, MMC, drawdown, and missing/late rounds without changing either model.
- Report preliminary per-round scores, but wait for multiple resolved rounds before judging
  leaderboard competitiveness.

Gate: both slots produce accepted unstaked submissions. Any replacement requires a versioned
artifact and explicit authorization; model history is permanent.

## Automation policy

Model Upload is the primary scheduler: Numerai runs the fixed CPU callable when live data is
released, so no personal computer needs to remain online. GPU training is a one-time production
refit for this experiment. Periodic retraining is deliberately deferred until the fixed models
have accumulated interpretable live history.

Monitoring may be automated read-only. Automated staking, deleting model slots, changing model
configuration, or uploading replacement artifacts is out of scope.

## User actions

Completed: the account exists; the unstaked `eden_eve` and `eden_adam` slots exist; and the
dedicated API public ID and secret are stored outside Git in Bitwarden. Remaining user action:
explicitly authorize the first two uploads. The secret must never be pasted into chat or committed.

## Success criteria

- Operational: both frozen models submit automatically and unstaked for every eligible round.
- Scientific: paired live scores exist for the same rounds and are evaluated without asymmetric
  intervention.
- Go: spectral has credible positive standalone CORR and compares favourably with AdamW across
  multiple resolved rounds, with BMC/MMC treated as separate contribution endpoints.
- No-go: persistent non-positive CORR, repeated operational failure under official limits, or a
  clear paired disadvantage after enough resolved rounds to distinguish regime noise.
