# Mentor Review

## Overall Assessment

This is one of the more carefully constructed plans I have reviewed: the question is genuinely open (the prior project's own weak-signal failure mode and Feldman's long-tail theory predict the opposite outcome from the MNIST result), the verdict definitions have teeth, and the fail-fast decomposition honestly concentrates uncertainty in the science rather than the infrastructure. Its biggest strength is that both outcomes are pre-framed as deliverables with mechanistic diagnostics that distinguish "didn't help" from "never engaged." Its biggest weakness is a missing structural separation between the eras used for hyperparameter tuning and the eras on which the verdict is computed — at a 0.005-corr verdict threshold, selection-induced optimism on shared eras is large enough to invalidate the headline comparison, and the arm with more degrees of freedom (the Spectral Optimizer, with its threshold modes and batch-size interaction) benefits asymmetrically.

## What a Senior Researcher Would Do Differently

1. **Separate tuning eras from verdict eras explicitly.** Component #8 tunes AdamW "on purged validation eras," and nothing in the plan reserves a distinct, untouched era set for the final paired comparison. Matched trial counts equalize *effort*, not *overfitting capacity*: the Spectral arm has extra hyperparameters (threshold mode hard/soft/variance, batch size interacting with the B×B matrix), so evaluating the verdict on the same eras used for selection biases toward the filter even under a matched budget. Fix: split validation-side eras into a tuning block and a final-verdict block (temporally later, with embargo), or use nested era-purged CV. This costs nothing — Numerai has hundreds of eras and the plan already notes era count is nearly free.

2. **Pre-register the threshold-mode decision rule.** The engagement gate (#1) passes if "at least one threshold mode" engages, and the success criteria say "try the soft/variance modes before concluding." Left as written, this is a garden of forking paths: three modes × selection-on-outcome quietly inflates the positive verdict. Either fix the mode at the engagement gate (choose the one with sane engagement diagnostics, *before* seeing out-of-sample performance) or declare mode a tuned hyperparameter inside the matched budget, evaluated only on tuning eras. State which, in the experiment plan, before any verdict run.

3. **Add a permutation-null calibration to the engagement diagnostic.** The MP threshold's null assumes i.i.d. noise, but per-sample gradients on financial data are cross-sectionally correlated within an era (shared factor exposure). The quant-finance RMT literature this project leans on warns exactly this: correlated data shifts the empirical bulk edge, so "eigenvalues above the MP bound" is not automatically signal. A cheap control fits inside the existing debug job: run the same diagnostic with permuted targets (pure noise by construction) and compare spectra. Without it, the pivot finding "no gradient-coherent signal separable from the MP bulk" — or its opposite — is not clearly interpretable.

4. **Make batch composition an explicit design decision.** Whether a batch draws samples from one era or many eras changes what "consensus" means: mixed-era batches let the filter keep cross-era common directions (plausibly the generalizable signal), while within-era batches make the consensus dominated by that era's common shock. This interacts directly with the FNC metric (which penalizes feature-exposure overfitting). Choose one, justify it, and log engagement diagnostics under it; a one-off comparison of the two in the #1 diagnostic run would be cheap and informative.

## What Hasn't Been Examined Yet

- **The tuning/verdict era contamination above** — easy to skip past because "era-purged CV" sounds like it covers it; it covers leakage between train and evaluation, not between selection and evaluation.
- **The non-i.i.d. MP null** — the plan transposes MP cleaning from returns covariance to gradient covariance but has not asked whether the MP null distribution survives the transposition on *this* data's correlation structure.
- **A cut-order inconsistency**: success-criteria.md designates the GAF ablation as a cut *before* the sequence arm is questioned ("cut this before the first two", with Muon/GBT as the pressure cuts), while the decomposition's #4 cut order is Muon → GAF → sequence arm. Given the sequence arm's P=0.35 and the novelty assessment's statement that the GAF ablation is "the single strongest way to convert related work into a sharp contribution," I would invert it: GAF ablation outranks the sequence arm. Resolve the inconsistency explicitly before Step 7.
- **Minor**: the "helps" bar mixes a CI criterion with an alternative "clear corr-Sharpe improvement" — the Sharpe route has no stated threshold or test and could become an escape hatch. Tie it to the same paired-CI machinery or drop it.

## Simpler Alternatives

The scoping is already close to minimal for a credible answer: single dataset, single architecture, filter-on/off at identical config as the primary evidence, sequence arm demoted to a stretch goal. I see no simpler path that still clears the tuned-AdamW/matched-budget credibility bar — a default-Adam comparison would be strictly cheaper and strictly worthless, which the plan already recognizes. One genuinely cheap addition worth considering (not required): a random-subspace control — project gradients onto a random subspace of the same dimension the filter keeps — which would distinguish "MP-selected directions matter" from "any gradient dimensionality reduction acts as a regularizer here." The GAF ablation partially covers this, but the random control is nearly free and sharper.

## Construct Validity / Information Value

The construct is sound. This is not a covert-behaviour topic, and the analogous checks pass: (1) the outcome is not statable on paper — the prior project's own evidence points in *both* directions (label-noise robustness on MNIST vs. degradation on weak-signal sparse parity), and Feldman's long-tail theory gives a principled mechanism for harm in exactly this regime, so the result is genuinely uncertain; (2) the evaluation object is faithful — real obfuscated financial data with era-purged temporal splits, per-era correlation, and a leakage sanity band, not a synthetic proxy; the plan even rejects the synthetic-label-flip proxy explicitly; (3) the plan answers the motivating question — it commits to a three-way verdict (helps/hurts/doesn't help) with an equivalence-style null rather than a neutral characterization, and the degenerate-filter outcome is pre-converted into a mechanistic finding rather than an uninterpretable shrug. The engagement diagnostics are precisely the instrument that keeps a null informative. The one validity risk is not the construct but the inference: the tuning/verdict era overlap identified above, which is fixable without redesign.

## Key Recommendations

1. **Reserve a final-verdict era set untouched by any tuning or mode selection** (or use nested era-purged CV); recompute power analysis (#3) on that set's era count. This is the load-bearing fix.
2. **Pre-register the threshold-mode decision rule** (fixed at the engagement gate, or tuned within the matched budget on tuning eras only) and add the permuted-target null-spectrum control to the #1/#7 debug job.
3. **Resolve the GAF-vs-sequence-arm priority inconsistency in favour of the GAF ablation**, and make batch composition (within-era vs mixed-era) an explicit, logged design choice.

## Verdict

MAJOR_REVISIONS

The science, scoping, and success criteria are strong, and the construct passes every information-value check — no rethink is warranted. But the absence of a tuning/verdict era separation is a load-bearing gap: with a 0.005-corr decision threshold and an asymmetric hyperparameter surface, computing the verdict on selection-contaminated eras could manufacture or mask the headline effect. It is a cheap structural fix, and it must be in place before any verdict experiment runs; the pre-registration of the threshold-mode rule and the permutation-null calibration should land at the same time.
