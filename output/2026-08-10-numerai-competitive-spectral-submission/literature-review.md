# Literature and Platform Synthesis

## Answer to “what number is comparable?”

No single historical score is exactly comparable to the current live leaderboard. The leaderboard is a one-year average of resolved forward submissions and currently exposes separate CORR20v2, MMC, BMC and CORJ60 reputations. Its payout target (`target_cyrus_20`) is not present in the released v5.3 historical files. The round-1329 generic historical `target` instead equals `target_ender_60` exactly.

We can nevertheless produce a rigorous two-level comparison:

1. **Offline:** exact official per-era CORR and contribution scores on frozen v5.3 walk-forward predictions, always naming the released scoring target, against the current released Ender benchmark on identical rows. This is comparable to Numerai Diagnostics and the official benchmark, not reputation.
2. **Forward:** after explicit upload authorization, parallel unstaked frozen submissions. Their resolved CORR20v2/MMC/BMC/CORJ60/Season are directly comparable to the current leaderboard.

## Experimental consequences

- Replace the provenance-free old shard with the current v5.3 Parquet snapshot and immutable hashes.
- Keep stocks from an era together. Use expanding historical folds and the official 8-era/16-era purges for 20D/60D targets.
- Use the 574 training eras for all HPO. Seal the 644 currently resolved official-validation eras until AdamW and spectral procedures are frozen.
- Train/evaluate named 20D target candidates relevant to current live Cyrus-20 (`target_cyrusd_20`, Ender-20 and a predeclared target ensemble) and the current generic Ender-60 target as a separate Diagnostics/CORJ60 track. Never silently pool their scores.
- Use exact `numerai-tools` metrics and version-matched `v53_lgbm_ender20/60` predictions.
- Treat era blocks—not millions of stock rows—as the statistical units.
- Give AdamW a genuine multi-fidelity HPO and the spectral procedure an equal recorded budget. The spectral method here is a streaming eigenspace of the p-by-p gradient covariance; FFT/parameter-frequency literature is not relevant evidence.
- Evaluate standalone models and frozen benchmark blends. A model with positive, stable BMC may be submission-worthy even if its standalone CORR is below the benchmark.

## Evidence quality and remaining uncertainty

Official docs/source determine platform facts and formulas. Academic work strongly supports nested temporal selection and shows that optimizer rankings reverse under unequal search spaces. Community reports support walk-forward embargoes and warn that repeated Diagnostics use overfits, but offer no reliable conversion from offline score to live rank and no controlled AdamW comparison. The prior project gives implementation evidence for this covariance-eigenspace filter, but its old data and metric cannot establish current Numerai quality.

The decisive residual uncertainty is empirical: whether a carefully tuned MLP supplies useful signal or benchmark-orthogonal contribution on v5.3, and whether spectral filtering improves that tuned MLP. The only decisive current-leaderboard test is prospective.
