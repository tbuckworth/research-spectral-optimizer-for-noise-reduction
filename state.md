---
run_id: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
topic: "Spectral Optimizer redo: pxp filter, walk-forward split, rank sweep"
current_step: 8
status: failed
fail_fast_agreement: true
criteria_approved: true
mode: autonomous
agent_backend: claude
agent_model: "fable"
issue_number: 2080
compute_profile: "MATS Slurm cluster, driven REMOTELY over SSH from wherever this run is executing (laptop or desktop) - you are NOT on the cluster. Every cluster command is prefixed: ssh mats '<cmd>' for the dev/login node, ssh mats-controller '<cmd>' for the controller (the only place gpu-avail and gpu-cost exist). AUTHORIZED COMPUTE: the FREE 'compute' partition only - one shared always-on node with 8x NVIDIA L40 (48GB VRAM each). Your concurrent cap across all your jobs is 6 GPUs, 124 CPUs, and 384GB RAM; max wall time is 24h per job. Request 1 GPU (--gres=gpu:1) unless the experiment genuinely needs more, and up to 6 when it does (single node, so torchrun/FSDP works). Sizing on 48GB: fp16 inference up to ~20B params, LoRA/QLoRA fine-tuning up to ~7B (13B with care), full fine-tuning only up to ~2-3B. Add --qos=debug for validation runs under 2h to jump the queue. PAID elastic-* partitions (A100/H100) are NOT authorized and the account is not enabled for them: if an experiment needs more than 6 L40s or more VRAM per device, do not attempt it - record a FAIL-on-affordability with the exact resource ask and continue. WORKFLOW for each experiment: (1) write the code and an sbatch script locally under experiments/exp-NNN/; (2) stage it with rsync -avP experiments/exp-NNN/ mats:/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN/; (3) submit with ssh mats 'cd /mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN && mkdir -p logs && sbatch run.sbatch' - create logs/ BEFORE sbatch, because Slurm opens the output file at job launch and an in-script mkdir is too late; (4) poll with ssh mats 'squeue -u t.buckworth' until the job leaves the queue (PD pending, R running, CG completing), sleeping 60s between polls rather than busy-looping - the free partition is shared and a PD (Resources) wait of tens of minutes is normal, NOT a failure, so keep waiting; do NOT end the step while a job you submitted is still queued or running - if you must stop, first record the job id and its experiment in state.md so the next attempt resumes waiting instead of resubmitting; and before submitting anything, run squeue and check whether a job for this experiment is already queued or running, because the step may be retried after an interruption and duplicate submissions waste the shared partition; (5) pull results back with rsync -avP mats:.../exp-NNN/ experiments/exp-NNN/ and copy the full slurm log into run.log. NEVER run training, fine-tuning, heavy inference, or long CPU loops in an ssh shell on the dev node - it is the shared login node and that is the cluster's worst etiquette violation. Never run agents or jobs on the controller. The local machine this run executes on is for orchestration, plotting, and light analysis of returned results only - do not silently fall back to a local GPU for experiments under this profile. Job script requirements: all #SBATCH lines must precede the first command (any command above them silently disables every directive below); include --partition=compute, --gres=gpu:1, a realistic --time, --job-name, --cpus-per-task=8, --mem=32G, --output=logs/slurm-%j.out. Storage on the cluster: code, checkpoints, and final results under /mnt/nw/home/t.buckworth (persistent NFS, NOT backed up - pull anything irreplaceable back to this machine); HuggingFace cache, dataset shards, and intermediate outputs under /ephemeral/t.buckworth (fast local scratch, wiped on reboot) via HF_HOME=/ephemeral/t.buckworth/hf. Inside every job script: source ~/venv/bin/activate (the cluster's shared venv) and run python with -u so logs are not buffered. That venv pins torch 2.5.1+cu121 to match the workers' CUDA 12.2 driver - do NOT upgrade torch or install one from default PyPI, which breaks CUDA on every job. Diagnostics: ssh mats 'scontrol show job <jobid>' explains a stuck job; sacct works only from the controller (it errors with Connection refused on the dev node); nvidia-smi on the dev node fails because there is no GPU there, which is expected; on compute, nvidia-smi inside a job lists all 8 physical GPUs but you only own $CUDA_VISIBLE_DEVICES; an empty .out file on a running job is usually stdout buffering. scancel anything left idle. Prefer lightweight experiments (small open-weight models), each targeting under 30 min of GPU time. Max 5 experiments."
knowledge_base: "none"
is_followup: true
parent_issue: 2073
prior_repo: https://github.com/tbuckworth/research-spectral-optimizer-for-noise-reduction
prior_run_id: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
followup_focus: from_decomposition
novelty_verdict: NOVEL
challenge_verdict: MINOR_REVISIONS
challenge_outcome: proceed_with_notes
design_triage: written
challenge_loop_count: 1
construct_validity: ok
clarifications:
  - q: "What is the motivating question (in the topic's own framing)?"
    a: "This is a follow-up to run 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri. The motivating question is unchanged and directional: does the Spectral Optimizer improve out-of-sample predictive performance versus tuned AdamW on large, noisy financial time-series data (Numerai v5)? The prior run found a rigorous, audited negative — but on the wrong optimizer (the per-sample B×B SpectralConsensusFilter instead of the actual p×p SpectralGradientFilter) and on a broken temporal split (~5.4-year train→test gap, no refit, baseline at 27% of Numerai's example model), so the answer does not count. The user's feedback: redo it correctly — same question, correct object, correct protocol. This run must return WINS / HURTS / NULL per the pre-registered decision rule: (B−A) 95% CI excluding zero with the same sign on >= 2 of 3 walk-forward folds."
  - q: "What do key terms mean?"
    a: "'Spectral Optimizer' = SpectralGradientFilter at the REPO ROOT of ~/pyg/optimizers in spectral_filter.py (verified present 2026-08-03: class at line 52, filter_grad() at line 303; repo pulled, up to date). It keeps a streaming rank-k factorization of the p×p gradient covariance (p = parameter count), updated by rank-1 SVD each step, projecting the batch-mean gradient onto top-k eigendirections; never materializes p×p; only a (k+1)×(k+1) eigh on CPU; ~2× a bare Adam step. Usage: filt = SpectralGradientFilter(model, base_opt, rank=k); loss.backward(); filt.filter_grad(); base_opt.step(). Key knobs: rank, decay, warmup, weighting (hard/soft), alpha, soft_residual, energy_threshold, adaptive (none/effrank/gap), normalize — use normalize='none' per prior finding H7. DO NOT USE experiments/spectral_optimizer.py (SpectralConsensusFilter, the B×B per-sample variant): prior work already recorded its H4 failure, and the parent's fp64-verified proof that its engagement eigenspectrum is exactly target-independent for scalar-output MSE is fatal — for that variant only. That theorem depends on row normalization and does NOT bind SpectralGradientFilter; do not re-derive or transfer it; cite it only as a scoped side-note. 'Walk-forward split' = 3-fold expanding-window: TRAIN (all usable eras to T) / embargo E / VALID (hp selection only) / embargo E / TEST, then REFIT from scratch on train+embargo+valid at VALID-selected hps before evaluating TEST once. E = ceil(20/5) = 4. 'Rank sweep' = the retained-eigendirection count is a hyperparameter tuned on VALID (fixed log grid ~{8,32,128,512,2048} capped by p, adaptive effrank/gap, energy_threshold {0.90,0.99}, soft alpha {0.5,1,2}), never a frozen operating point — freezing one from a diagnostic proxy was the parent's fatal process error."
  - q: "What is the implanted/target construct, and is it a faithful goal (not a fixed string)?"
    a: "N/A — empirical optimizer-evaluation study, not a covert-behaviour topic. The analogous validity requirements, all mandated by the feedback: (1) THE assertion of the run, hard-asserted and printed per fold: min(test_eras) - max(refit_train_eras) == E + 1 == 5 — the test period begins one embargo after training ends, preventing the parent's 5.4-year-gap failure; (2) hard baseline gate: tuned AdamW must reach >= 0.60× the Numerai example model's mean per-era corr on each fold's TEST eras (from data/v5.0_validation_example_preds.parquet), else the comparison does not run on that fold and remaining budget goes to fixing the baseline; (3) matched tuning budget: 12 trials per arm, arm B's search space includes learning rate; (4) the norm/k-matched random-subspace control (arm C) separates 'spectral selection matters' from 'any low-rank projection does this'; (5) alpha=0 soft identity must reproduce plain AdamW, plus a seeded zero-predictor control. With these, the outcome is genuinely uncertain in both directions."
  - q: "What prior work is known?"
    a: "The parent run's 28-reference literature base carries over unchanged (copied to literature/; references.bib and citation-registry.md copied): closest neighbor Gradient Agreement Filtering (Chaubard et al. 2024); RMT/Marchenko-Pastur covariance cleaning in finance (returns covariance, not training gradients); Feldman long-tail memorization as counter-hypothesis; Gorishniy et al. matched-tuning-budget methodology. Parent-run empirical priors that inform THIS run: H7 (raw covariance beats var/degree normalization), H3 (rank <= 4 destabilizes; ~10 fastest on sparse parity), H5 (adaptive effective-rank was the only parity-grokking variant — 'adaptive spectral rank helps iff the solution is genuinely low-dimensional'), parent audit Finding 5 (fp32 cuSOLVER eigh fails stochastically under rank collapse — wrap the per-step eigh in a CPU-fp64 fallback from the start, log and report every firing; working patch in parent audit/rerun-exp-004/src/), and the parent's measured per-era corr autocorrelation (recompute per fold, do not inherit). Parent's exp-004 negative is a fact about the B×B variant on a broken split — 'Not Worth Pursuing' items from parent next-steps.md remain binding."
  - q: "What does success look like?"
    a: "A verdict on the motivating question delivered against a baseline that demonstrably works (gate >= 0.60× example model passed per fold), with rank tuned on VALID rather than guessed, on a split where TEST begins one embargo after the refit training data ends. Decision rule: WINS if (B−A) > 0 with 95% CI excluding zero on >= 2 of 3 folds same sign; HURTS if the mirror; NULL otherwise; effect sizes with CIs reported regardless. Mechanism honesty: if B beats A but is statistically indistinguishable from C (random subspace), the conclusion is 'low-rank projection helps; spectral selection is not the active ingredient'. KILL CRITERION: if the gate passes, the rank sweep genuinely ran at matched budget, and the result is NULL or HURTS on >= 2 of 3 folds, the line is dead for financial tabular regression — write the clean negative and STOP; no fourth epicycle, no new dataset, no full-scale-rerun rescue. A well-evidenced negative counts only if the gate passed and the sweep actually happened. Pre-register the PROTOCOL (splits, embargo E=4, metrics, moving-block bootstrap, decision rule, baseline gate) before touching TEST; do NOT pre-register a filter operating point. Metrics: primary mean per-era numerai_corr on TEST; secondary mean per-era Spearman and corr-Sharpe (fixed stated bootstrap RNG, stability checked across >= 2 RNG seeds); >= 3 paired seeds per arm per fold; report (B−A), (B−C), (C−A) per fold with 95% moving-block-bootstrap CIs, block length from per-fold lag-1 ACF. [Step 6 challenge added binding amendments to the kill criterion and sweep spec — see decisions step 6.]"
  - q: "What is the scope and compute profile?"
    a: "Experiments run within the run's compute profile: MATS Slurm free 'compute' partition (L40s), driven remotely over SSH, 1 GPU per job, jobs targeting under ~30 GPU-min, max 5 experiments; do not exceed it — components needing more become future work. The p×p filter is ~2× Adam and parent AdamW runs were ~3 s, so this fits; the cost drivers are the full v5.0 feature set (mandatory unless VRAM forces reduction — then document as limitation) and longer re-tuned schedules on the much larger expanding-window training sets (parent's 2000 steps @ B=1024 must NOT be inherited). If a fold will not fit, reduce seeds before folds, and say so. Data (~6.3 GB) already on local disk in the parent run's data/ — copy or symlink, DO NOT re-download. Scope: MLP is the primary and probably only architecture; DO NOT rebuild the GRU arm (reshaped-tabular construct); 'Stadia/RushCursive' garble is ignored; no live leaderboard submission — walk-forward TEST blocks are the leaderboard equivalent. Reuse parent code: exp-001/src (download, data_prep era-purge/embargo arithmetic), exp-003/src (shard build, AdamW sweep harness), exp-004/src (random-subspace control, per-era eval, moving-block bootstrap), audit/rerun-exp-004 (independent bootstrap + eigh-fallback patch). Realized fold boundaries written to protocol.json (V ~= 96, S ~= 110, 3 contiguous TEST blocks covering the most recent data, coverage >= 0.95, computed from the realized usable-era list)."
  - q: "What assumptions are being made?"
    a: "(1) SpectralGradientFilter integrates as documented (filter_grad() filters .grad in place between backward() and step()) — checked by the free alpha=0 identity test before any comparison is trusted; (2) the expanding-window training sets fit L40 VRAM/time budgets with the full feature set at re-tuned step counts; (3) the 0.60× example-model baseline gate is reachable with more data + refit + full features (the parent's shortfall was protocol-induced, not intrinsic) — if not, budget goes to baseline fixes (features, steps, network size, LR schedule, target transform, regularization) before any comparison; (4) per-fold moving-block bootstrap with recomputed block length gives valid CIs for era-autocorrelated per-era corr; (5) the (k+1)×(k+1) eigh fallback keeps numerics sound (parent audit Finding 5); (6) rank-1 SVD covariance updates remain well-conditioned over the longer schedules. All challenged in Step 6 — see challenge/ and the step 6 decision for outcomes."
decisions:
  - step: 1
    decision: "Follow-up scope: from_decomposition. The feedback changes the research design (p×p SpectralGradientFilter instead of B×B variant; 3-fold expanding-window walk-forward with refit instead of the broken gap split; rank swept on VALID instead of frozen; baseline gate; matched 12-trial tuning) but not the framing, literature, or novelty. Copied from parent: literature/ (4 files), novelty-assessment.md, references.bib, citation-registry.md. Regenerating: success-criteria.md (Step 4), decomposition.md (Step 5), challenge/ (Step 6)."
  - step: 1
    decision: "Deliberate deviation from the from_decomposition default of current_step: 4 — set current_step: 3 so Step 4 REGENERATES success-criteria.md instead of copying the parent's. Rationale: the parent's criteria are anchored to the B×B variant and the broken split, and success-criteria.md is the frozen anchor for the Step 10 audit; reusing it would audit this run against the defects the run exists to fix. The brief's baseline gate (>= 0.60× example model), decision rule (>= 2/3 folds, CI excluding zero), matched tuning budget, and walk-forward protocol are the new criteria's core and are binding on the criteria agent."
  - step: 1
    decision: "Mandatory precondition verified before anything else (per the brief): git -C ~/pyg/optimizers pull → already up to date; ~/pyg/optimizers/spectral_filter.py exists and exposes SpectralGradientFilter (line 52) with filter_grad() (line 303). No file substitution needed."
  - step: 1
    decision: "novelty_verdict: NOVEL carried from parent — the contribution remains the transfer verdict (does gradient-covariance spectral filtering help low-SNR financial regression), now measured on the correct object; the niche was unoccupied across 28 references and nothing in the feedback changes the novelty position. The parent's negative is scoped to the B×B variant and does not pre-answer this run (its theorem depends on row normalization, which SpectralGradientFilter does not do). [Step 6 caveat: parent A4/F11 classified the B×B→p×p swap as a scope change requiring a novelty re-check against GaLore-style low-rank gradient projection, K-FAC lineage, and momentum-subspace methods; a targeted mini-search is REQUIRED before write-up (Step 11) — risk is to positioning, not experimental validity.]"
  - step: 1
    decision: "Binding constraints carried into all downstream steps: (a) hard per-fold assert min(test_eras) - max(refit_train_eras) == 5, printed in every fold's log — a run without this assertion firing is not answering the question; (b) baseline gate is a hard gate — no optimizer comparison on a fold whose gate fails; (c) kill criterion — gate passed + genuine matched-budget rank sweep + NULL/HURTS on >= 2/3 folds ⇒ clean negative, STOP, no rescue proposals; (d) DO NOT rebuild the GRU arm; (e) cite the parent's target-independence proof only as a scoped side-note about the B×B variant; (f) CPU-fp64 eigh fallback with logged firing count from the start; (g) protocol pre-registered before TEST is touched, filter operating point NOT pre-registered."
  - step: 1
    decision: "No knowledge base for this run (knowledge_base: none) — all KB actions skipped silently per workflow. Parent has no prior/knowledge/ directory (predates repo KBs); parent next-steps.md read directly from the parent run dir instead — this brief is its ranked step #1."
  - step: 4
    decision: "success-criteria.md REGENERATED from scratch (not copied from parent) and auto-approved (autonomous mode). It is the frozen Step 10 audit anchor. All 10 binding elements from the follow-up brief are embedded: directional WINS/HURTS/NULL deliverable; pre-registered decision rule (95% CI excluding zero, same sign, >= 2 of 3 folds); hard baseline gate (arm A >= 0.60x example model per-fold TEST mean per-era corr, no comparison on a failed-gate fold); per-fold hard assert min(test_eras) - max(refit_train_eras) == E + 1 == 5 with E=4; matched 12-trial budget with LR in arm B's space and rank swept on VALID (never frozen); arm C norm/k-matched random-subspace mechanism control with the honesty clause; alpha=0 identity + zero-predictor sanity controls; metrics spec (per-era numerai_corr primary, Spearman + corr-Sharpe secondary, >= 3 paired seeds, per-fold moving-block bootstrap with recomputed block length, RNG fixed and stability-checked across >= 2 seeds, (B-A)/(B-C)/(C-A) per fold); kill criterion (clean negative = success, valid only if gate passed and sweep ran); protocol pre-registered before TEST, filter operating point NOT pre-registered. Notable design choices: an 'uninterpretable outcome' definition (failed gate on >= 2 unfixable folds, truncated sweep, assertion never fired) so the audit can distinguish a clean negative from a broken run; compute feasibility as a binding section (Muon baseline, second architecture, extra folds = future work, not criteria, and their absence is not an audit finding); the 5-experiment cap read as submitted experiment units with short trials batched inside jobs — Step 5 must size the decomposition to this. SOTA reference points: example model +0.0235 on parent test eras; parent AdamW +0.0064 (below its own sanity band, protocol-induced); parent B-x-B negative scoped as a side-note only. [SUPERSEDED by Step 6 loop-back: regeneration round 2 must preserve ALL of the above AND add the A-series amendments from the step 6 decision.]"
  - step: 5
    decision: "Decomposition REGENERATED for the corrected design (agent read prior/decomposition.md, all 5 prior results.md files, and the follow-up brief). Parent PASS infrastructure banked as reuse, not re-tested: data (#9 marked SKIP — symlink from parent data/, never re-download), era arithmetic, sweep harness, bootstrap machinery, eigh-fallback patch (src/ from parent run dir or repo clone since prior/experiments/ holds only results/plans). 8 active components; no SHOWSTOPPERS. Dominant genuine uncertainty is #5 the baseline gate (P=0.55, untested assumption 3) — plan is structured to buy that bit early: fold 1 runs alone, gate-first (A sweep → refit → single TEST eval → gate ratio + the ==5 assertion) before folds 2-3 are spent. New risk surfaced: rank-2048 grid point's per-step (k+1)x(k+1) CPU eigh is cubic in k — the '~2x Adam' figure is unverified there; pre-flight debug job measures s/step across the whole rank grid and sets the realized grid (capping = documented design realization under the brief's 'feasible for p' clause, not a failure). Experiment-unit packing honors the criteria's sizing note: EU-1 pre-flight debug (throughput/VRAM/rank grid/identity re-assert), EU-2 fold 1 gate-first, EU-3/EU-4 folds 2-3 in parallel after fold-1 gate passes, EU-5 reserve for baseline-fix retry. Cut order if overflow: cap rank grid → reduce seeds (floor 3) → reduce folds, reported. P_success ~0.09 full verdict deliverable, ~0.23 any informative outcome (gate failure converts to the criteria's defined baseline-construction negative). Binding: a structural alpha=0 identity failure (#3) is STOP-and-report per the brief, not a component to engineer around; kill criterion means no rescue components exist downstream of a clean NULL/HURTS. [To be REGENERATED after Step 6 loop-back with the B-series amendments from the step 6 decision.]"
  - step: 6
    decision: "Challenge complete: three independent passes dispatched in parallel (assumption-analysis.md: 13 assumptions, 5 critical; mentor-review.md: verdict MAJOR_REVISIONS; pre-mortem.md: 5 scenarios, gate-unreachable Critical). CONSTRUCT-VALIDITY GATE: ok — mentor-review explicitly verified the outcome is not statable a priori (parent's target-independence theorem correctly scoped away from SpectralGradientFilter; literature predicts both directions), the object is the real p×p filter via its documented API, and the plan returns a committed WINS/HURTS/NULL verdict; no strawman, no redesign loop. VERDICT MAPPING: MAJOR_REVISIONS with challenge_loop_count 0 → loop back once; challenge_loop_count set to 1 (no further challenge loops permitted). DELIBERATE DEVIATION from the executor default (current_step: 4 → Step 5 only): current_step set to 3 so Step 4 ALSO re-runs, because the convergent load-bearing defects live in success-criteria.md — the frozen Step 10 audit anchor (missing F12 guard, unqualified kill criterion, unspecified trial allocation). Re-running decomposition alone cannot fix the anchor; this mirrors the documented step-1 precedent that auditing against defective criteria is wrong. BINDING AMENDMENTS — Step 4 regeneration round 2 must preserve every element listed in the step 4 decision AND add (A-series, criteria/protocol): A1 power-qualified kill: NULL is terminal only if the realized per-fold MDE (from the fold's own moving-block bootstrap) <= 0.005 on the qualifying folds, else the verdict is 'NULL (power-limited): |B−A| bounded within ±MDE; line not declared dead' with a resourced decisive-test ask in Future Work; per-fold MDE reported next to every CI as a standing table. A2 kill-scope floor: the kill verdict is auto-scoped to the realized rank grid; if realized K_max < 512, NULL/HURTS reports 'no evidence at feasible ranks' plus a costed future-work ask instead of killing the line; adaptive arms' realized k(t) logged so 'the cap binds the adaptive arms' is checkable from artifacts. A3 F12 under-exploration guard reinstated as binding: pre-registered signatures (arm-B best config on grid boundary; non-monotone/high-variance sweep across rank; sharp arm-B optimal-LR shift vs arm A) downgrade a HURTS to 'no evidence of benefit under the affordable tuning budget'. A4 pre-registered arm-B trial allocation: staged design over a PRUNED space (stage 1: one trial per rank/adaptive point at transferred LR, centered on H3/H5; stage 2: 4-5 trials refining LR/alpha around the stage-1 winner); pruning (e.g. drop 'gap' or one energy_threshold) documented under the brief's feasibility clause; converged step count is NOT inside the 12 trials — fixed by the EU-1 full-length arm-A convergence run; refit stopping rule (e.g. steps scaled by rows_refit/rows_train) pre-registered in protocol.json. A5 gate semantics: example-model denominator computed LOCALLY per prospective TEST block before protocol freeze (free — parent out/example_per_era_corr.csv); degenerate-yardstick fallback pre-registered now (denominator floor ~0.010 → absolute floor or all-validation-mean ratio); near-miss band: realized ratio in [0.50,0.60) → one targeted fix then accept; < 0.45 → structural gap, reserve buys diagnosis + data-scaling learning curve (converts to evidence on assumption 3), not repair; pre-specified baseline-fix ladder with per-rung trial budgets and a stopping rule (regularization/network size first per parent F12 signatures); gate retries select on the gate ratio ONLY, comparison hps re-selected on VALID after any baseline fix, TEST-touch count logged per fold and reported; gate fixes restricted to p-preserving levers unless EU-1 also timed arm B at the larger architecture; era-recency WEIGHTING declared an authorized gate-fix lever compatible with the no-subsampling mandate. A6 inference spec: seed-variance handling in the bootstrap CI stated explicitly (hierarchical/seed-block resampling, or a reported check that seed variance << era variance); paired seeds raised to 5 if EU packing permits (the only upward power lever); fold non-independence acknowledged in claim language ('consistent across 3 nested folds covering [dates]', not independent replications) with cross-fold correlation of per-era (B−A) reported. A7 arm C re-derived for parameter space: matching invariants are k(t), update-norm ratio, AND the basis-rotation rate — preferred design: random orthogonal rotation of B's own realized basis (matches everything except spectral identity of directions); ~20-min local CPU sim (captured-energy fraction + realized norm-amplification at k in {8,512,2048}, p=600k) REQUIRED before porting code; C reported in absolute terms (own loss curves + TEST scores); mechanism-honesty clause made symmetric (B >> C claim requires fair-control evidence, else downgrades to 'adaptive low-rank projection helps'). A8 verification depth: alpha=0 identity supplemented by a planted-subspace correctness check (hard top-k recovers a planted dominant direction) and per-step filtered-vs-unfiltered cosine + kept-norm-fraction logging in all arm-B runs; selected config's distance-from-identity reported per fold; assert every TEST era of every fold present in example preds, alongside THE ==5 assertion with the raw-era↔usable-index mapping recorded in protocol.json; decay/warmup fixed at repo defaults declared a limitation NOW (or one grid dimension swapped for decay in {0.99,0.999}). B-series (Step 5 re-decomposition): B1 EU-1 additionally runs one FULL-LENGTH arm-A convergence run (real steps-to-convergence + proxy gate ratio on a VALID slice — collapses the ~500x-extrapolated cost anchor and buys the gate bit one unit earlier), times arm B at the largest plausible gate-fix architecture (or the p-preserving restriction of A5 applies), and times the eigh-every-N-steps and GPU-fp32-eigh variants (named documented variants if ever used). B2 fold jobs made resumable BEFORE any submission: per-trial result persistence to NFS (verify parent exp-003 harness append survives), refit checkpoints, --time at 2-3x projection, folds 2-3 re-projected from fold-1 REALIZED wall not EU-1 numbers; fold projecting > ~16h → split into sweep-job + refit/eval-job. B3 arm C component redesigned per A7 with the CPU sim as its quick test and a basis-rotation-rate (principal-angle) diagnostic logged in B runs. B4 component #2 power sim: if P(CI excludes 0 | true effect ±0.005) < 0.6 on >= 2 prospective folds, A1's power-qualified wording is load-bearing and MUST be in the frozen protocol (it is mandatory regardless). C-series (Step 11 write-up notes, carried forward): C1 targeted novelty mini-search (GaLore-style low-rank gradient projection, momentum-subspace/low-pass optimizers 2024-2026, K-FAC lineage, applied to regression/finance) before write-up; C2 mechanism described as TEMPORAL mean-gradient-subspace filtering (EMA over steps), not per-sample consensus — interpretive claims gated on the A8 diagnostics. Accepted risks (no design change): fold correlation (claim calibration only), interpretive-frame transfer (write-up gating), torch 2.5.1 compat (EU-1 re-assert covers it)."
  - step: 4
    decision: "ROUND 2: success-criteria.md regenerated after the Step 6 MAJOR_REVISIONS loop-back and auto-approved (autonomous mode). Replaces round 1 as the frozen Step 10 audit anchor. All round-1 binding elements preserved in force; amendments A1-A8 integrated in place as binding criteria: A1 power-qualified kill (NULL terminal only if per-fold MDE <= 0.005 on >= 2 gated folds, else 'NULL (power-limited)' with a resourced decisive-test ask; MDE reported as a standing table next to every CI); A2 kill-scope floor (realized K_max < 512 => 'no evidence at feasible ranks' + costed ask, not a kill; adaptive arms' realized k(t) logged); A3 F12 under-exploration guard with operationalized signatures (arm-B best config on grid boundary; non-monotone sweep with range < ~2x across-seed sd; >= 4x optimal-LR shift vs arm A) downgrading HURTS to 'no evidence of benefit under the affordable tuning budget'; A4 staged arm-B allocation over a pruned space (stage 1: one trial per rank/adaptive point at transferred LR centered on H3/H5, 7-8 trials; stage 2: 4-5 trials refining LR/alpha around the winner; pruning documented: drop 'gap' and energy_threshold 0.99; converged step count fixed by the EU-1 full-length arm-A convergence run, NOT inside the 12; refit stopping rule steps scaled by rows_refit/rows_train pre-registered in protocol.json); A5 full gate semantics (example-model denominator computed locally per prospective TEST block before protocol freeze; degenerate-yardstick fallback pre-registered as a denominator floor at 0.010 with a low-signal flag — the 'absolute floor' option, chosen over all-validation-mean for monotonicity; bands: [0.50,0.60) one targeted fix then accept, [0.45,0.50) fix ladder, < 0.45 structural gap => reserve buys diagnosis + data-scaling learning curve; 5-rung fix ladder with per-rung trial budgets 4/4/2/1/1 and a stopping rule, regularization/network-size first; retries select on gate ratio ONLY, comparison hps re-selected on VALID after any fix, TEST-touch count logged per fold; p-preserving levers only unless EU-1 timed arm B at the larger architecture; era-recency weighting authorized); A6 inference spec (hierarchical era-block x seed bootstrap as headline CI with negligibility-check alternative reported; paired seeds target 5 if packing permits, floor 3; nested-fold claim language, cross-fold correlation of per-era (B-A) reported); A7 arm C re-derived for parameter space (matching invariants k(t) + update-norm ratio + basis-rotation rate; preferred design: random orthogonal rotation of B's own realized basis; ~20-min local CPU sim REQUIRED before porting code; C reported in absolute terms; symmetric mechanism-honesty clause); A8 verification depth (planted-subspace correctness check; per-step filtered-vs-unfiltered cosine + kept-norm-fraction logging in all arm-B runs; selected config's distance-from-identity per fold; example-preds coverage assertion for every TEST era beside THE ==5 assertion; raw-era<->usable-index mapping in protocol.json; decay/warmup at repo defaults declared a limitation NOW, with Step 5 permitted to swap one pruned grid dimension for decay in {0.99,0.999}). Verdict taxonomy extended so the qualified forms (power-limited, scope-limited, budget-limited, baseline-construction negative) are pre-registered outcomes the audit can check claims against. Next: Step 5 re-decomposition under the B-series amendments."
  - step: 5
    decision: "ROUND 2: decomposition.md REGENERATED after the Step 6 MAJOR_REVISIONS loop-back (agent handed the round-1 file as base, all three challenge/ files, the round-2 frozen criteria, prior decomposition + prior experiment results, and the B1-B4 amendment list). Replaces round 1, which it records as superseded. All four B-series amendments visibly implemented and indexed in a change-log table: B1 — EU-1 expanded with one FULL-LENGTH arm-A convergence run (real steps-to-convergence + VALID-slice proxy gate ratio, buying the gate bit one unit before fold 1), arm-B timing at the fix-ladder rung-2 architecture (~2.5-3M params; without it A5's p-preserving restriction applies and rung 2 is skipped), and eigh-every-N / GPU-fp32-eigh variant timings (named documented variants if ever used); packing arithmetic (#1) re-anchored on measured steps-to-convergence instead of the ~500x-extrapolated parent 3s anchor. B2 — new component #7 fold-job resumability BEFORE any submission: kill-test of the parent exp-003 harness's append survival, refit checkpoints including filter state, --time at 2-3x projection, folds 2-3 re-projected from fold-1 REALIZED wall, >~16h folds split into sweep-job + refit/eval-job with the reserve absorbing the extra submission. B3 — arm C (#2) rewritten from mechanical port to re-derived control: invariants k(t) + update-norm ratio + basis-rotation rate; preferred design random orthogonal rotation of B's own realized basis (with the honest caveat a Haar-rotated basis may still capture ~k/p energy — exactly what the REQUIRED ~20-min local CPU sim at k in {8,512,2048}, p~600k decides before any code is ported); principal-angle diagnostic logged in all arm-B runs; round-1's fixed-subspace fallback explicitly retired as unacceptable (norm-matched noise injection). B4 — component #3's fail branch wired, not narrative: A1 power-qualified kill wording goes into frozen protocol.json regardless; sim FAIL (P(detect +/-0.005) < 0.6 on >= 2 prospective folds) makes it load-bearing for the write-up; P for #3 lowered 0.85 -> 0.70 per the pre-mortem's ACF evidence. A-series touchpoints landed: per-fold gate denominators + 0.010 floor + example-preds coverage assertion in #9; planted-subspace check + cosine/kept-norm/distance-from-identity logging in #5; staged 12-trial A4 allocation, 5-seed A6 target (floor 3), and an 11-item protocol-freeze checklist gating EU-2. Delegated decay/warmup decision EXERCISED: one stage-2 trial runs the stage-1 winner at decay=0.999 (warmup stays repo default, remains a declared limitation). Preserved from round 1: gate-first fold-1, EU-1..EU-5 packing, cut order (cap rank grid -> seeds floor 3 -> folds), alpha=0 structural failure = STOP-and-report, no rescue downstream of a clean kill, data SKIP. 9 active components + 1 SKIP; no SHOWSTOPPERS. P ~0.06 unqualified verdict (down from round 1's 0.09 — honestly so: round 1 counted outcomes the challenge showed could be artifacts), ~0.23 any informative outcome (unchanged — amendments converted those failure shapes into pre-registered qualified findings). Dominant uncertainty remains #4 the baseline gate (P=0.55), bought earliest via EU-1 proxy ratio then gate-first fold 1."
  - step: 6
    decision: "ROUND 2 challenge complete: three fresh independent passes dispatched in parallel against the round-2 artefacts (round-1 files preserved as challenge/round1-*.md). Amendment verification requested from each: assumption-analysis confirms all twelve amendments A1-A8/B1-B4 are genuinely implemented in the documents (located file-by-file in its table), and mentor-review independently confirms the same defect-by-defect; the round-2 P drop 0.09 -> 0.06 is judged honest accounting. CONSTRUCT-VALIDITY GATE: ok - mentor-review re-verified the headline outcome is not statable a priori (filter never run on this data; parent theorem correctly scoped away; literature predicts both directions), the object is the real p-x-p filter via its documented API, and the plan returns a committed verdict; the one baked-in-outcome risk found (a degenerate arm C - noise or clone - would predetermine the mechanism sub-claim in either direction) is cheaply fixable inside the plan (D2) and does not trigger redesign. construct_validity stays ok; construct_loop_count untouched. VERDICT MAPPING: mentor-review verdict MINOR_REVISIONS -> proceed with notes (challenge_loop_count already 1; no further loops available and none needed). challenge_outcome: proceed_with_notes. FINDINGS SNAPSHOT: assumption-analysis 13 assumptions (4 critical - risk migrated from missing machinery into what the new machinery assumes: arm-A-defined step count fair to arm B; a fair arm C exists between the noise and clone degeneracies; MDE-as-half-width is ~50-percent-power and 0.005 is ~35 percent of the gated baseline; transferred LR ranks the rank grid); pre-mortem 5 scenarios (gate-unreachable-and-proxy-miscalls High/High Critical, EU-1 amendment-cargo collapse High/Medium, freeze-locked LR defect self-triggering A3 Medium/High, clean-run triple-hedged non-answer High/Medium systemic, arm-C noise-or-clone Medium/Medium; cross-cutting: single-shot inputs feed an irreversible freeze, interpretability bought but not decisiveness, both new early-warning instruments uncalibrated - both calibrations free). LIMITATION TRIAGE (design-time, per shared rubric): written to challenge/limitation-triage.md - 13 fix-now-free items D1-D13 (every one a zero-GPU pre-freeze wording/logging/sim-extension/job-structuring change: 4th under-exploration signature + matched-step-budget claim language; arm-C distinctness + planted-task discriminability + empty-middle fallback; MDE derivation/renaming/P-detect linkage + CI coverage check; kept-norm-scaled stage-1 LR + LR-probe trial; refit-extension rule; EU-1 phase-persistence with convergence run last + EU-1b fallback charged to reserve; proxy calibrated with example-model VALID-vs-TEST offset + reserve pre-commitment; pooled cross-fold secondary estimand + pre-freeze decisiveness checkpoint; signature-2 sd source + trial-count reconciliation; flagged-fold role pre-registration; post-hoc invariant-match report; kill-scope instantiation axes; protocol.json as single operational spec with per-job checklist logging + decay-probe priority rule) - ALL folded into decomposition.md as the binding Round-2 Challenge Addendum (D-series), extending the protocol-freeze checklist to 16 items; 4 future-work rows FW1-FW4 (era-level power ceiling with a costed decisive-test ask, dataset/architecture/scale generality, rank-grid ceiling beyond K_max, decay/warmup study) carry to Step 11; accepted risks stated (gate reachability genuinely ~50/50 and only running tests it; calibrated proxy still heuristic; B4 sim input provenance; fold correlation as claim language; torch pin via EU-1 re-assert; C1 novelty mini-search stays deferred to Step 11). CAP ACCOUNTING: no fix adds an experiment unit; D4/D9 trials live inside the 12-trial budget per A4 slack; EU-1b and fold splits charge to the EU-5 reserve as already accounted - the 5-experiment cap stands. design_triage: written. Next: Step 7 (report planned experiments)."
  - step: 8
    decision: "Fail-fast agreement confirmed (trivial in autonomous mode): fail_fast_agreement set true. Binding on Step 9: a FAIL result stops further experiments per the run-level bite points recorded at Step 7 (structural alpha=0 identity failure -> STOP-and-report; EU-1 infeasibility -> FAIL-on-affordability with exact resource ask; fold-1 terminal gate failure -> pre-registered baseline-construction negative, folds 2-3 gate-first phase only if failure mode is plausibly fold-specific). Note the plan's own structure already encodes fail-fast ordering: EU-1 before the 16-item protocol freeze, freeze before EU-2, EU-2 alone before EU-3/EU-4, EU-5 reserve on pre-registered triggers only."
  - step: 7
    decision: "Experiment plan auto-approved (autonomous mode) from the round-2 decomposition. Rules applied: (1) SKIP removal — component #10 (Numerai v5 data + example preds, P=0.97, banked by parent) excluded from the plan; its action (symlink + row/era sanity check) is a Wave-1 setup task, not an experiment. (2) Cap — the plan counts cluster submissions per the criteria's binding sizing note: exactly 5 experiment units (EU-1 pre-flight, EU-2 fold 1 gate-first, EU-3 fold 2, EU-4 fold 3, EU-5 reserve), at the cap with zero headroom; contingency submissions (EU-1b convergence-run continuation per D6, B2 fold splits) are pre-authorized charges AGAINST the EU-5 reserve, not additions — if the reserve is consumed by one contingency it is not available for another, and the cut order (cap rank grid -> seeds floor 3 -> reduce folds) applies before any thought of a 6th unit. (3) rethink_disproof: not set — full plan stands. The 6 zero-GPU local components (#9 fold arithmetic + protocol.json + denominators, #5 filter integration + identity + planted-subspace, #7 resumability kill-test, #3 power sim + MDE, #2 arm-C design sim, #1 packing arithmetic) are prerequisites inside the plan, not units — Waves 1-3 of the parallelisation plan, all gating the 16-item protocol freeze that must complete before EU-2 touches TEST. Execution-order constraints binding on Step 9: EU-1 before freeze; freeze before EU-2; EU-2 (fold 1) runs ALONE and its realized gate ratio + wall time size EU-3/EU-4; EU-3/EU-4 submit only after fold-1 gate passes (parallel, 2 GPUs, within cap); EU-5 spends only per its pre-registered triggers (A5 ladder bands / structural-gap learning curve / forced rerun / D6-D7-B2 absorptions). Fail-fast semantics per unit recorded in the plan table; the run-level fail-fast bite points are: alpha=0 structural identity failure (STOP-and-report), EU-1 infeasibility (FAIL-on-affordability with exact resource ask), fold-1 gate terminal failure after band-appropriate ladder spend (baseline-construction negative — folds 2-3 still run their OWN gate-first phase 1 only if fold-1's failure mode suggests fold-specific rather than structural causes; a structural < 0.45 diagnosis redirects EU-5 to the learning curve and the run reports the pre-registered negative without spending folds 2-3). No reordering: lambda order within waves is preserved from the decomposition; the EU sequence is dependency-forced. current_step: 7, status: experiments_planned."
experiment_plan:
  - unit: EU-1
    name: "Pre-flight (--qos=debug): throughput/VRAM/realized rank grid + B1 additions"
    components: "#8 (lambda 0.18), #6 short-schedule check (lambda 0.22), #5 on-cluster alpha=0 re-assert"
    tests: "Full-scale feasibility: shard build; s/step for arms A and B at rank {8,32,128,512,2048} at planned AND rung-2 gate-fix architectures; eigh-every-N + GPU-fp32-eigh variant timings; VRAM peaks; 500-step stability + fallback count; D6 ordered persisted phases with the FULL-LENGTH arm-A convergence run LAST (checkpointed, VALID-score streamed, plateau stop) yielding measured steps-to-convergence + D7-calibrated proxy gate ratio; adaptive k(t) + rotation-rate logging"
    pass: "Arm A trains at full scale; convergence run plateaus (or EU-1b continuation per D6, charged to EU-5); realized grid keeps >= 4 points incl >= 512 (else A2 wording pre-registered immediately); VRAM <= 44GB; no unexplained NaN; second-architecture + variant timings recorded"
    fail: "Full feature set unfittable even with int8 residency + reduced batch (-> documented feature reduction, mandated limitation), or arm B > 10x arm A at every rank >= 32 (-> symmetric trial shrink, else FAIL-on-affordability with exact ask)"
    est_wall: "~1-2 h"
  - unit: EU-2
    name: "Fold 1, gate-first, B2-resumable"
    components: "#4 baseline gate (lambda 0.24, P=0.55 — the dominant genuine uncertainty), #6 long-schedule retirement"
    tests: "Phase 1 (unconditional): 12-trial arm-A VALID sweep -> refit under pre-registered stopping/extension rules -> single TEST eval -> realized gate ratio vs floored local denominator + THE ==5 assertion + coverage assertion printed. Phase 2 (conditional on gate, wired into sbatch per D7): A4 staged arm-B sweep (12 trials incl kept-norm LR probe D4 + decay=0.999 probe) -> refits A/B/C x paired seeds (target 5, floor 3) -> one TEST pass with all A8/D1 diagnostics"
    pass: "Gate: realized ratio >= 0.60 ([0.50,0.60) resolves per near-miss band: one targeted rung, one re-check, accept). Unit: phases complete with per-trial NFS persistence intact, assertions green, TEST-touch count logged"
    fail: "Ratio < 0.60 after genuine sweep + refit + band-appropriate ladder spend (per-band: [0.45,0.50) ladder within EU-5 budgets 4/4/2/1/1 + futility rule; < 0.45 structural -> EU-5 buys diagnosis + data-scaling learning curve, not repair). Terminal gate failure on >= 2 folds = pre-registered baseline-construction negative; no optimizer comparison on failed-gate folds"
    est_wall: "~3-6 h projected from EU-1 measurements; --time at 2-3x"
  - unit: EU-3
    name: "Fold 2 (submitted only after fold-1 gate pass)"
    components: "#4 continued; main comparison"
    tests: "Same gate-first two-phase structure as EU-2; sized from fold-1 REALIZED wall (B2), not EU-1 numbers"
    pass: "Own gate >= 0.60 and phases complete; contributes its fold to the >= 2/3 decision rule"
    fail: "Own gate failure handled per A5 bands; fold recorded as failed-gate for the decision rule's gated-fold accounting"
    est_wall: "re-projected from fold-1 realized wall"
  - unit: EU-4
    name: "Fold 3 (largest shard; parallel with EU-3)"
    components: "#4 continued; main comparison"
    tests: "Same structure; largest expanding-window shard; split into sweep-job + refit/eval-job if > ~16 h projected (B2, absorbing EU-5)"
    pass: "Own gate >= 0.60 and phases complete; verdict computable per pre-registered rule (four-condition kill: execution + A1 power + A2 scope + A3 under-exploration)"
    fail: "As EU-3; if a split is needed AND EU-5 is already spent, cut order applies (cap grid -> 3 seeds -> reduce folds, reported)"
    est_wall: "re-projected from fold-1 realized wall"
  - unit: EU-5
    name: "Reserve (conditional; pre-registered triggers only)"
    components: "contingency for #4/#8"
    tests: "One of: A5 fix-ladder retry on a near-miss/ladder-band fold; structural-gap diagnosis + data-scaling learning curve (converts gate failure into evidence on assumption 3); forced rerun; absorption of EU-1b (D6) or a B2 fold split"
    pass: "Reserve spent only on a pre-registered trigger, spend and trigger logged"
    fail: "N/A — unspent reserve is a fine outcome"
    est_wall: "<= 4 h"
local_prerequisites: "Zero-GPU, no unit cost, all before protocol freeze: Wave 1 [#9 fold arithmetic + protocol.json + THE assertion + gate denominators w/ 0.010 floor (P=0.85); #5 filter integration + alpha=0 fp64 identity + planted-subspace + eigh fallback + diagnostics (P=0.80, structural identity failure = STOP-and-report); #7 resumability kill-test + refit checkpoint roundtrip (P=0.85)]; then [#3 hierarchical-bootstrap power sim + per-fold MDE + B4 wiring (P=0.70); #2 arm-C B3/D2 design sim, REQUIRED before porting control code (P=0.65)]; Wave 3 after EU-1 [#1 packing arithmetic, seeds 5-vs-3 decided (P=0.70)]; then 16-item protocol freeze gating EU-2"
lambda_table:
  - component: "Design packs into <=5 experiment units (5-seed option, staged sweep, resumable structure)"
    p_success: 0.70
    t_hours: 0.25
    lambda: 1.43
    status: PENDING
  - component: "Arm C re-derived for parameter space (k(t), norm ratio, basis-rotation rate) [B3/A7]"
    p_success: 0.65
    t_hours: 0.75
    lambda: 0.57
    status: PENDING
  - component: "Per-fold inference power at ~110 TEST eras + MDE machinery (A6 hierarchical bootstrap) [B4]"
    p_success: 0.70
    t_hours: 0.75
    lambda: 0.48
    status: PENDING
  - component: "Baseline gate: tuned AdamW >= 0.60x floored example-model denominator (>=2 of 3 folds) [A5]"
    p_success: 0.55
    t_hours: 2.5
    lambda: 0.24
    status: PENDING
  - component: "SpectralGradientFilter integration + alpha=0 identity + planted-subspace + eigh fallback + A8 diagnostics"
    p_success: 0.80
    t_hours: 1.0
    lambda: 0.22
    status: PENDING
  - component: "Numerical stability over long re-tuned schedules"
    p_success: 0.80
    t_hours: 1.0
    lambda: 0.22
    status: PENDING
  - component: "Fold-job resumability + realized-wall projection [B2, new in round 2]"
    p_success: 0.85
    t_hours: 0.75
    lambda: 0.22
    status: PENDING
  - component: "Full-scale throughput/VRAM + realized rank grid + B1 additions (convergence run, 2nd architecture, eigh variants)"
    p_success: 0.70
    t_hours: 2.0
    lambda: 0.18
    status: PENDING
  - component: "Walk-forward fold arithmetic + protocol.json + THE assertion + gate denominators + coverage assertion [A5/A8]"
    p_success: 0.85
    t_hours: 1.0
    lambda: 0.16
    status: PENDING
  - component: "Numerai v5 data + example preds (banked by parent)"
    p_success: 0.97
    t_hours: 0.1
    lambda: 0.30
    status: SKIP
---

# Workflow Progress

## Step 1: Clarifications (follow-up)

Follow-up to 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
(parent issue #2073, this issue #2080).

The parent run found: a rigorous, independently audited HURTS/negative verdict
(−0.00527 mean per-era corr, CI [−0.00886, −0.00181], replicated on a GRU;
spectral selection indistinguishable from a random-subspace control) — but it
tested the wrong optimizer (per-sample B×B SpectralConsensusFilter, whose H4 had
already failed in prior work) against a broken temporal protocol (~5.4-year
train→test gap, no refit, baseline at 27% of Numerai's example model, below the
parent's own sanity band). The user's feedback: the answer does not count; redo
it correctly with the p×p SpectralGradientFilter, an expanding-window
walk-forward split with refit, a hard baseline gate, and a genuine rank sweep at
matched tuning budget.

This run focuses on: the same directional motivating question, executed on the
correct object under a correct protocol. followup_focus: from_decomposition —
literature/novelty copied from parent; success criteria, decomposition, and
challenge regenerated (current_step set to 3 so Step 4 rebuilds the criteria,
which are also the Step 10 audit anchor). Precondition verified:
SpectralGradientFilter present in ~/pyg/optimizers/spectral_filter.py after pull.

Artifacts: followup-summary.md written; literature/ (4 files),
novelty-assessment.md, references.bib, citation-registry.md copied from parent.
Next: wrapper runs Step 4 (success criteria).

## Step 4: Success Criteria (round 1 — superseded)

success-criteria.md regenerated by the criteria agent and auto-approved
(autonomous mode). This file is the frozen anchor for the Step 10 audit — it
replaces the parent's criteria entirely, which were anchored to the B×B
variant and the broken split. Full content summary in the step 4 decision
entry. SUPERSEDED: the Step 6 challenge found the regenerated criteria dropped
the parent's F12 under-exploration guard while making the kill criterion
terminal, left the 12-trial arm-B allocation unspecified against a ~42-point
space, and left the kill criterion unqualified by power (MDE) and rank-grid
scope. Round 2 must preserve all round-1 binding elements and add amendments
A1–A8 (step 6 decision).

## Step 5: Steinhardt Decomposition (round 1 — superseded)

decomposition.md regenerated (8 active components + 1 SKIP; no SHOWSTOPPERS;
gate-first fold-1 structure; EU-1 pre-flight / EU-2 fold 1 / EU-3+4 folds 2–3 /
EU-5 reserve; P_success ~0.09 full verdict, ~0.23 any informative outcome).
Full content summary in the step 5 decision entry. SUPERSEDED: to be
regenerated after the Step 6 loop-back with amendments B1–B4 (step 6
decision) — EU-1 gains a full-length arm-A convergence run and variant
timings, fold jobs become resumable with per-trial persistence, arm C is
redesigned for parameter-space semantics (basis-rotation-rate matching), and
the power sim's consequence is wired to the amended kill wording.

## Step 6: Challenge (round 1 — MAJOR_REVISIONS, loop back to Step 4)

Three independent passes dispatched in parallel and completed:
- challenge/assumption-analysis.md — 13 assumptions (5 critical): arm C's
  parameter-space port changes its semantics (k/p ≈ 0.3% captured energy →
  norm-matched noise injection, vacuous honesty clause); 12 trials cannot
  resolve the enumerated sweep space and F12 was dropped; the gate's
  denominator is an unexamined per-fold random variable; gate-fix levers are
  coupled to the treatment's feasibility envelope through p; the NOVEL verdict
  was carried across the exact object swap the parent flagged as needing a
  novelty re-check.
- challenge/mentor-review.md — verdict MAJOR_REVISIONS. Construct sound, no
  rethink; but the matched 12-trial budget cannot resolve the enumerated arm-B
  space as written, and there is no pre-registered downgrade path protecting
  the terminal kill criterion from an under-exploration artifact. All required
  fixes are zero-compute protocol amendments.
- challenge/pre-mortem.md — 5 scenarios: gate unreachable (High/High —
  ~half the failure mass, denominator check + fix ladder are free); NULL by
  power, killed by rule (Medium/High); feasibility-capped rank grid stops
  covering the hypothesis (Medium/High); atomic fold jobs die of logistics
  (Medium/Medium); arm C matches the wrong invariants (Medium/Medium).
  Cross-cutting: the kill criterion verifies execution, not evidential
  sufficiency — the parent's central failure one level up.

Construct-validity gate: ok (no known-outcome flaw; outcome genuinely
uncertain in both directions; committed verdict machinery). construct_validity: ok.

Decision: MAJOR_REVISIONS, challenge_loop_count 0 → 1. Loop back with a
documented deviation: current_step: 3 (not the executor default of 4) so the
wrapper re-runs Step 4 THEN Step 5, because the convergent defects live in
success-criteria.md — the frozen Step 10 audit anchor — which a
decomposition-only loop cannot fix. Binding amendment series A (criteria),
B (decomposition), C (write-up) recorded in the step 6 decision entry; the
Step 4 and Step 5 re-runs MUST hand their agents the three challenge/ files
plus the amendment list, and preserve all round-1 binding elements.
Limitation triage deferred to the post-loop Step 6 pass (per executor rules,
triage only runs when proceeding).

Next: wrapper runs Step 4 (criteria regeneration, round 2).

## Step 4: Success Criteria (round 2 — frozen audit anchor)

success-criteria.md regenerated by a fresh criteria agent handed the round-1
file as base, the three challenge/ files, the followup brief, and the A1-A8
amendment list; auto-approved (autonomous mode). The document reads as one
coherent pre-registration: every round-1 binding element survives in force,
and the A-series amendments are woven into the sections where they operate
(kill criterion now a four-condition rule: execution validity + A1 power
qualification + A2 rank-grid scope floor + A3 under-exploration downgrade;
gate semantics, sweep allocation, inference spec, arm C design, and
verification depth all amended as recorded in the step 4 round-2 decision
entry). Two authorized-alternative choices flagged by the agent: the
degenerate-yardstick fallback is the denominator-floor option (0.010), and
decay/warmup is declared a limitation now with Step 5 permitted to trade one
pruned grid dimension for decay in {0.99, 0.999}.

Next: wrapper runs Step 5 (decomposition regeneration, round 2, under the
B-series amendments B1-B4 from the step 6 decision).

## Step 5: Steinhardt Decomposition (round 2)

decomposition.md regenerated by a fresh decomposition agent handed the
round-1 file as base, the three challenge/ files, the round-2 frozen
criteria, the prior run's decomposition and experiment results, and the
B1-B4 amendment list. All four B-series amendments are visibly implemented
and indexed in a change-log table at the top of the file (B1: EU-1 gains a
full-length arm-A convergence run + VALID-slice proxy gate ratio, second
architecture timing, and eigh-variant timings; B2: new resumability
component #7 with kill-test, checkpoints, realized-wall re-projection, and
fold-split rule; B3: arm C re-derived for parameter space with the required
pre-port local CPU sim as its quick test; B4: the power sim's consequence
wired to the mandatory A1 kill wording in protocol.json). The decay/warmup
authorization is exercised as a decay=0.999 stage-2 probe; warmup remains a
declared limitation. An 11-item protocol-freeze checklist gates EU-2 (no
TEST touch before freeze). 9 active components + 1 SKIP (data banked by
parent); no SHOWSTOPPERS. Highest-lambda component: EU packing arithmetic
(1.43); dominant genuine uncertainty: the baseline gate (#4, P=0.55),
bought earliest via the EU-1 proxy ratio then gate-first fold 1. Overall
P ~0.06 for an unqualified verdict, ~0.23 for any informative outcome
(gate/power/scope failures convert to pre-registered qualified findings
under the round-2 criteria rather than uninterpretable runs). Full detail
in the step 5 round-2 decision entry.

Next: wrapper runs Step 6 (challenge, round 2 pass — construct gate,
verdict mapping with challenge_loop_count already at 1, and the
post-loop limitation triage).

## Step 6: Challenge (round 2 — MINOR_REVISIONS, proceed with notes)

Three fresh independent passes dispatched in parallel against the round-2
artefacts (round-1 challenge files preserved as challenge/round1-*.md):

- challenge/assumption-analysis.md — verifies all twelve A/B amendments are
  genuinely implemented (per-amendment location table), then finds 13
  assumptions (4 critical, 6 moderate, 3 background). The critical cluster is
  new in kind: round 1's defects were missing machinery; the residual risk now
  lives in what the machinery itself quietly assumes — an arm-A-defined step
  count read as fair to arm B, a fair parameter-space arm C existing between
  the noise and clone degeneracies, an MDE floor that is a ~50%-power
  half-width and ~35% of the gated baseline's own score, and a transferred LR
  assumed to rank the rank grid. Three of the four feed the same construct the
  kill criterion certifies ("genuine sweep at matched budget"). Every
  recommended fix is a zero-compute pre-freeze change.
- challenge/mentor-review.md — verdict MINOR_REVISIONS. Checked each round-1
  defect against the actual text: all fixed in binding form, none merely
  claimed. Construct validity re-verified sound (no known-outcome flaw). Three
  in-flight recommendations: pre-designate arm-C candidate (b) as expected
  primary, derive the 0.005 MDE anchor in protocol.json, protect EU-1 with a
  plateau stop and qos-escalation fallback. Flags specification-fidelity risk
  (the criteria are now the run's most complex artifact) with protocol.json-as-
  operational-spec as the mitigation.
- challenge/pre-mortem.md — 5 scenarios: gate unreachable + uncalibrated proxy
  (High/High, Critical; assumption 3 remains ~50/50 and the dominant untested
  premise); EU-1 collapsing under its amendment cargo (High/Medium; the
  convergence run's duration is the unknown EU-1 exists to measure);
  freeze-locked defect via the A4→A3 self-trigger (Medium/High); clean run
  delivering a triple-hedged non-answer (High/Medium; the systemic one —
  interpretability was bought, decisiveness was not, and the qualification
  triggers are correlated through the compute envelope); arm C noise-or-clone
  (Medium/Medium). Both new early-warning instruments (proxy ratio, C-sim
  acceptance test) are themselves uncalibrated; both calibrations are free.

Construct-validity gate: OK (re-verified; no redesign). Verdict mapping:
MINOR_REVISIONS → proceed_with_notes; challenge_loop_count stays 1.

Design-time limitation triage written to challenge/limitation-triage.md:
13 fix-now-free items (D1–D13, all zero-GPU pre-freeze changes) folded into
decomposition.md as the binding Round-2 Challenge Addendum, extending the
protocol-freeze checklist from 11 to 16 items; 4 future-work rows (FW1–FW4)
carry to Step 11; accepted risks stated plainly, headed by the genuinely
~50/50 baseline gate that only running can test. No fix adds an experiment
unit — the 5-experiment cap stands, with EU-1b/fold-split submissions
chargeable to the EU-5 reserve as already accounted.

Next: wrapper runs Step 7 (report planned experiments).

## Step 7: Planned Experiments (auto-approved)

The round-2 decomposition's experiment-unit packing is approved as the
experiment plan, unmodified — it already satisfies both Step 7 rules:

- **SKIP removal**: component #10 (Numerai v5 data, P=0.97, banked by the
  parent run) is excluded; its symlink + sanity check is Wave-1 setup, not an
  experiment.
- **Cap**: exactly 5 experiment units, where a unit = one Slurm submission
  batching many short trials (the criteria's binding reading of the cap).
  Zero headroom remains: EU-1b (D6 convergence continuation) and B2 fold
  splits are pre-authorized charges against the EU-5 reserve, not additions,
  and the reserve can absorb only one such contingency. Overflow beyond that
  triggers the preserved cut order (cap rank grid → 3 seeds → reduce folds),
  never a sixth unit.
- `rethink_disproof` is not set; the full plan stands.

Approved units (full pass/fail detail in the `experiment_plan` frontmatter):

| Unit | Content | Components (λ) | Gate/fail-fast semantics |
|------|---------|----------------|--------------------------|
| EU-1 | Pre-flight `--qos=debug`: rank-grid throughput at 2 architectures, eigh variants, VRAM, stability, on-cluster α=0 re-assert, D6-phased full-length arm-A convergence run + D7-calibrated proxy gate ratio | #8 (0.18), #6 short (0.22), #5 guard | Infeasibility → documented reduction or FAIL-on-affordability; K_max < 512 → A2 wording pre-registered immediately |
| EU-2 | Fold 1, gate-first, B2-resumable: arm-A sweep→refit→TEST gate (phase 1); staged arm-B sweep + A/B/C refits + TEST (phase 2, conditional in sbatch) | #4 (0.24, P=0.55), #6 long | THE ==5 assertion + coverage assertion; gate < 0.60 handled per A5 bands; terminal failure = baseline-construction negative |
| EU-3 | Fold 2, after fold-1 gate pass, sized from realized wall | #4 | Own gate per A5 |
| EU-4 | Fold 3, parallel with EU-3, largest shard, split rule if > ~16 h | #4 | Own gate per A5 |
| EU-5 | Reserve: ladder retry / structural-gap learning curve / forced rerun / EU-1b or fold-split absorption | contingency | Spent only on pre-registered triggers, logged |

Six zero-GPU local components precede any fold submission (Waves 1–3:
#9, #5, #7, #3, #2, then #1 after EU-1), all feeding the 16-item
protocol-freeze checklist that gates EU-2. Run-level fail-fast bite points,
binding on Step 9: structural α=0 identity failure → STOP-and-report;
EU-1 infeasibility → FAIL-on-affordability with the exact resource ask;
fold-1 terminal gate failure → the pre-registered negative, with folds 2–3
spent only if the failure mode is plausibly fold-specific rather than
structural (< 0.45 diagnosis redirects EU-5 to the data-scaling learning
curve instead).

Next: wrapper runs Step 8 (fail-fast agreement — trivial in autonomous mode).

## Step 8: Fail-Fast Agreement (confirmed)

`fail_fast_agreement: true` recorded. Step 9 executes the approved 5-unit plan
under fail-fast semantics: on any unit-level FAIL matching the run-level bite
points (structural α=0 identity failure; EU-1 infeasibility; fold-1 terminal
gate failure), stop further experiments, record the result, and set
`status: experiments_done_with_failure` so the audit triages genuine-null vs
botched-run. Conditional/qualified outcomes that the round-2 criteria
pre-register (gate near-miss bands, power-limited NULL, scope-limited grid) are
NOT fail-fast stops — they are handled inside the plan's own branching.

Next: wrapper runs Step 9 (execute experiments).

## Step 9: Execute Experiments (IN PROGRESS)

Session resumed 2026-08-04. Found: prior Step 9 session was interrupted after
writing plans for exp-f01..exp-f04 (local zero-GPU components #9, #5, #7, #3) —
no results existed, no cluster jobs queued (squeue empty), no run dir staged on
the cluster. Data symlinks verified intact.

Experiment-directory convention (follow-up numbering): exp-f01=#9 fold
arithmetic, exp-f02=#5 filter integration, exp-f03=#7 resumability,
exp-f04=#3 power sim (waits on f01), exp-f05=#2 arm-C design sim (plan written
this session), exp-f06=EU-1 pre-flight (cluster), exp-f07=EU-2 fold 1,
exp-f08=EU-3 fold 2, exp-f09=EU-4 fold 3, exp-f10=EU-5 reserve (if triggered).

Progress log:
- exp-f01 (#9 fold arithmetic): PASS. All 40 assertions green on 3 folds. THE
  ==5 assertion holds (gaps 896-891, 1006-1001, 1116-1111); V=96/S=110 exact;
  TEST blocks 0896-1005, 1006-1115, 1116-1225, coverage 1.0; no era gaps so
  raw<->usable mapping is identity shift. Denominators +0.03957/+0.02756/
  +0.01639, all above 0.010 floor (D10: recorded, no flags). D7 T/V ratios
  1.166/0.678/0.572 — uncalibrated proxy would OVERSTATE TEST on folds 2-3;
  calibrated rule pre-registered. Downstream notes: example-preds parquet has
  no era column (id-join needed); filter out data_type=test eras 1226-1230;
  example-model edge decays ~2.4x across TEST blocks (fold 3 nearest floor).
  protocol-draft.json written (append-ready for freeze).
- exp-f03 (#7 resumability, B2): PASS. Kill-test (real SIGKILL mid-trial-4):
  completed trials survive via O_APPEND+fsync, restart runs exactly the
  remainder, no duplicates. Checkpoint roundtrip incl. filter state (V, S,
  proj_k, step_count, grad_mean + 4 RNG streams; filter has no state_dict —
  helpers written in src/checkpoint.py): bitwise fp64 continuation over 10
  steps. Resume entrypoint harness.py demonstrated end-to-end. FINDINGS:
  parent exp-003 sweep.py buffered in memory — would NOT have survived a
  --time kill (B2 was a real fix); fold jobs do NOT need sweep/refit split —
  resumable single-submission design stands for EU-2/3/4. Deliverables:
  src/trial_store.py, checkpoint.py, harness.py.
- exp-f02 (#5 filter integration): PASS, all 5 sub-checks + RNG separation.
  alpha=0 identity BIT-IDENTICAL over 50 fp64 steps (no STOP); planted-subspace
  cosine 0.977; fallback fires+logged (count 3, forced); zero-predictor null
  exact 0. Deliverables: src/spectral_filter.py (canonical copy, "RUN-COPY MOD
  1" = fallback patch), src/diagnostics.py, src/numerai_eval.py. Interface
  notes for cluster jobs: gram/eigh dtype follows torch DEFAULT dtype, eigh
  CPU-resident even on GPU; filter_grad() reassigns p.grad each step (no
  caching); energy_threshold=0.9 + adaptive rules collapse to k=1-2 on
  top-heavy spectra; no torch-2.11-only APIs (2.5.1-safe).
- EU-1/exp-f06 LAUNCHED (cluster agent; all prerequisites in hand).
- RUNNING: exp-f04 (power sim), exp-f05 (arm-C sim), exp-f06 (EU-1 cluster).
- PENDING: EU-1/exp-f06 (plan written; launch after f02+f03); packing
  arithmetic #1 + 16-item protocol freeze (needs EU-1); EU-2..EU-5.
- Cluster jobs submitted so far: NONE.

### Resume 2026-08-04 ~11:25 (third Step 9 session) — reconciliation + relaunch

Reconciled the previous entry's in-flight claims against disk and cluster:
- exp-f04 (power sim): process died with the prior session BEFORE writing any
  output (log stalled at "launching power_sim.py", out/ empty). RELAUNCHED via
  a fresh experiment agent (code exists; smoke tests had passed).
- exp-f05 (arm-C sim): part1_stream FULL run (T=2400) died at t=1800, no
  process alive; part2_descent never ran and carries 3 recorded defects.
  Agent resuming: clean part1 restart + part2 with the three fixes.
- exp-f06 (EU-1): NO driver script or run.sbatch existed; NO job submitted
  (state's "LAUNCHED" reflected an agent that died before writing code).
  Found DUPLICATE data rsyncs (10:53 and 10:59, both mid-validation); killed
  the older (which also lacked example_preds), kept the 10:59 survivor —
  verified progressing (~36% of v5.0_validation at ~5MB/s at 11:22).
  train.parquet + features.json staged and size-verified on the cluster.
  EU-1 agent launched: waits for staging to finish + size-verifies all 4
  files, writes phase driver + run.sbatch (D6 per-phase persistence),
  submits spectral-eu1 (--qos=debug, 1 GPU), polls at 60s.
- Cluster occupancy: 3 UNRELATED jobs from other projects (shade-vllm-preflight,
  exp007 x2) hold 3 of the shared 6-GPU cap — 3 GPUs headroom, EU-1 needs 1.
  These jobs are not this run's and must never be scancel'd.

In-flight now: exp-f04 agent, exp-f05 agent, exp-f06/EU-1 agent (all parallel,
independent). After EU-1 returns: #1 packing arithmetic + 16-item protocol
freeze, then EU-2 (fold 1, gate-first, alone), then EU-3/EU-4, EU-5 per
triggers. Cluster jobs submitted so far: NONE (EU-1 submission pending agent).
