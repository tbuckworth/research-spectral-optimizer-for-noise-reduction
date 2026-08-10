# Mentor Review

## Overall Assessment

The plan is unusually careful about temporal leakage, metric parity, target naming, and the distinction between historical Diagnostics and live reputation. Its largest weakness is that the proposed optimizer comparison is not actually fair under the user's intended estimand: it uses wall-clock/accelerator-time constraints and an asymmetric promotion funnel, while also wavering between a fixed-architecture optimizer ablation and a comparison of fully tuned training procedures.

## What a Senior Researcher Would Do Differently

First, define the estimand before running anything. If the question is whether replacing AdamW's raw gradient with the spectral filter helps a strong AdamW pipeline, freeze the architecture, data order, number of examples, optimizer-update count, folds, seeds, scheduler horizon, and post-processing, then tune only optimizer-specific hyperparameters with the same number of completed configurations at the same fidelities. If the question is instead whether the best spectral training procedure beats the best AdamW procedure, both arms need the same architecture search envelope and completed-configuration budget. The current plan claims the latter in the success criteria but implements the former by freezing AdamW-selected architecture and non-optimizer choices.

Second, remove runtime from scientific fairness and branch-continuation rules. Component 3's `<=3x` overhead threshold, component 9's accelerator-time matching, the “whichever binds first” rule, and the 60-versus-120 accelerator-hour allocation all conflict with the explicit constraint that spectral overhead is irrelevant. Feasibility should test only whether a configuration can finish within available memory and the project horizon. Fairness should be audited by completed configurations, training examples or optimizer updates per configuration, fold evaluations, seeds, and identical fidelity levels. Failed runs caused by numerical or algorithmic instability should remain outcomes; infrastructure-preempted runs should be rerun and should not consume one arm's scientific budget.

Third, make the search funnel symmetric. AdamW currently receives a 30–40-configuration multi-fidelity search, while spectral must first beat the already selected AdamW using three settings and a `0.0005` threshold before receiving its nominal search. That tests whether an almost untuned spectral method immediately beats a tuned control, not whether equally tuned procedures differ. Apply the same predeclared successive-halving schedule to both arms, or use a paired design in which every AdamW base configuration has a matched spectral configuration and optimizer-specific parameters are allocated an equal number of completed trials.

Fourth, simplify and predeclare the confirmatory endpoint. Choose one primary 20D target and one primary metric for the optimizer claim, with CORR delta as stated, and treat Ender-20, the target ensemble, Ender-60, BMC, Sharpe, drawdown, neutralization, and blends as secondary analyses. Otherwise the many target/post-processing routes create substantial researcher degrees of freedom despite the seal. The freeze manifest should include an explicit multiplicity policy and a single rule for choosing among target tracks.

Fifth, distinguish model selection uncertainty from era-level uncertainty. A block bootstrap over eras conditional on the selected configurations does not account for selection over dozens of noisy HPO trials. Use repeated outer train-only folds or nested walk-forward selection: select each arm inside each outer fold, then compare their predictions on that fold's held-out eras. The sealed official validation can remain the final confirmation, but the development estimate should evaluate the complete selection procedure rather than only the winning checkpoints.

## What Hasn't Been Examined Yet

The plan has not resolved whether the spectral basis is global across all parameters, layer-wise, or tensor-wise, although that choice changes both the mechanism and the meaning of a rank-matched random-subspace control. It also does not specify whether weight decay is applied before or after projection, whether Adam moments are built from raw or projected gradients, or whether clipping precedes covariance estimation. These are load-bearing implementation choices, not minor engineering details, and strength-zero equivalence alone will not identify them.

The comparison may be confounded by unequal effective optimization. Matching nominal updates is necessary, but spectral projection can change gradient norm and therefore effective step size. The plan should log update norms, cosine similarity to the raw gradient, retained gradient energy, basis turnover, and Adam moment norms. Add a norm-matched control that rescales the raw AdamW update to the spectral update norm; otherwise an apparent spectral effect may be ordinary step-size attenuation.

The random-subspace control is underspecified. A fresh random subspace each step, a fixed random subspace, and a subspace with the same temporal persistence and update cadence test different hypotheses. The most diagnostic control matches rank, update cadence, persistence, and retained-energy distribution while randomizing orientation. A shuffled-gradient-history basis is also valuable because it tests whether temporal alignment, rather than low rank alone, carries the effect.

The AdamW gate “beat the original recipe by 0.001” is not evidence that AdamW is competitive and should not determine whether the spectral question is scientifically answerable. The old recipe is weak and historically incomparable by the plan's own account. A valid optimizer comparison can proceed with the strongest honestly selected AdamW baseline even if HPO improves it by less than an arbitrary amount; competitiveness against Ender is a separate gate.

Likewise, “95% of benchmark CORR” is an arbitrary competitiveness threshold, and positive BMC plus a statistically positive blend increment can be achieved by a very small blend weight with negligible practical value. Predeclare a minimum economically or operationally meaningful blend improvement and report the selected blend weight. Benchmark-relative claims should be based on nested OOF blend selection, not selection and evaluation on the same OOF predictions.

Finally, the prospective plan does not state the minimum number of resolved live rounds or a live stopping rule. Preparing an artifact is in scope, but any later leaderboard-quality claim needs a predeclared observation window and frozen submission behavior; otherwise the project can wait or stop opportunistically.

## Simpler Alternatives

The most direct scientific experiment is a paired optimizer study on one fixed, already credible MLP architecture and one primary 20D target. Run an equal number of completed AdamW and spectral configurations under an identical multi-fidelity schedule; match examples, updates, folds, batch streams, and seeds; select each arm using nested train-only walk-forward folds; then reveal the official validation once. Report standalone CORR delta as the confirmatory result and treat BMC, mechanism diagnostics, random/shuffled subspaces, and benchmark blends as secondary analyses.

This removes the 60D track, model-upload packaging, neutralization search, and broad blend search from the critical path. Those can follow only if the optimizer comparison or the strongest AdamW model is promising. A lightweight tree benchmark is also useful as a data-and-scoring sanity check, but it should not become another tuned competitor unless the objective expands beyond the MLP optimizer question.

## Construct Validity / Information Value

The headline result is not determined by construction. A learned persistent gradient subspace could improve, harm, or leave unchanged out-of-era performance, so the experiment has genuine information value. The construct is also faithful to the stated mechanism: it tests a streaming gradient-covariance eigenspace filter rather than an FFT proxy, and it can return a clear verdict on whether that filter improves a tuned AdamW MLP on released Numerai targets.

The current execution plan nevertheless weakens that verdict because the two procedures receive asymmetric selection opportunities and fairness partly depends on runtime. Under the present rules, a spectral failure could mean either “the method does not help” or “three lightly tuned settings did not beat a fully tuned AdamW model before the full search was allowed.” This is a design defect, but it is repairable without changing the core construct.

## Key Recommendations

1. Replace all wall-clock and accelerator-time fairness criteria with matched completed configurations, examples or updates, folds, seeds, and fidelity levels; retain runtime only as a feasibility/logistics measure.
2. State one estimand and implement it consistently, using a symmetric search and promotion schedule for AdamW and spectral rather than screening three spectral settings against the tuned winner.
3. Predeclare one primary 20D target and endpoint, use nested walk-forward evaluation of the full selection procedure, and demote target ensembles, 60D, post-processing, blends, and packaging from the optimizer experiment's critical path.

## Verdict

MAJOR_REVISIONS

The core research question and construct are sound, and the leakage controls are strong. However, the current fairness accounting and asymmetric search funnel are load-bearing flaws: they can produce an uninterpretable negative result and directly violate the user's stated constraint. Correct those before spending substantial compute.
