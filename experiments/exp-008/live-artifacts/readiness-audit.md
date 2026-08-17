# Numerai live-candidate readiness audit

Audited 2026-08-17 against Numerai round 1334. This is an evidence summary; the
large immutable model, fixture, prediction, and JSON audit artifacts are retained
under `/mnt/nw/home/t.buckworth/numerai-live-candidates/results/bounded-live-bundles`.

## Frozen candidates

| Slot | Training arm | Ensemble | Parameters per model | Callable SHA-256 |
|---|---|---:|---:|---|
| `eden_adam` | AdamW, config 38 | seeds 0, 1, 2 | 6,799,361 | `4817ffe32b813ee901d089ea2df99667de909420e7a5f005d39213461b37a717` |
| `eden_eve` | spectral, config 39 | seeds 0, 1, 2 | 10,194,433 | `d57791c58883fc6cc84137a2ef66841bbf42df81d9b9fd6a9cded1883668bf7d` |

All six models completed 20,000 updates over the 2,746,268-row v5.3 production
training shard. The refit audit passed with freeze SHA-256
`ad823a6a4cbca262ef5f16190cd20941572142bd8af5adcfcdd2e54541c2b63c`.

## Current-round validation

The round-1334 fixture contained 7,008 IDs. Its target-labelled schema columns
were all null; the downloader rejects the fixture if any target value is revealed.

| Candidate | Local CPU time | Peak RSS | Official time | Official drift | Result |
|---|---:|---:|---:|---:|---|
| AdamW | 11.89 s | 1.16 GB | 29 s | 0.0 | pass |
| spectral | 12.97 s | 1.20 GB | 15 s | 0.0 | pass |

Both exact pickles passed the one-CPU, 4 GB, 600-second contract. The official
Python 3.12 runner at commit
`2b6eb43612db25bcf047608b4c1fd61cc3ff3c06`, image
`sha256:26969634731da15d1f8688ca3d4b62882cda69b0280a3c0f4b6089c295a1f797`,
reproduced both conventional prediction files exactly (correlation 1.0 and
maximum absolute difference 0.0).

The complete experiment test suite passed: 191 tests.

## Authorization boundary

These artifacts are technically ready for an unstaked live upload. No prediction
or model has been uploaded, and no staking action has been taken. Upload and
staking remain separate actions requiring explicit user authorization.
