---
run_id: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
topic: "Spectral Optimizer (for noise reduction) on Financial Timeseries Data"
current_step: 11
status: complete
mode: autonomous
agent_backend: claude
agent_model: "fable"
issue_number: 2073
compute_profile: "MATS Slurm cluster, driven REMOTELY over SSH from wherever this run is executing (laptop or desktop) - you are NOT on the cluster. Every cluster command is prefixed: ssh mats '<cmd>' for the dev/login node, ssh mats-controller '<cmd>' for the controller (the only place gpu-avail and gpu-cost exist). AUTHORIZED COMPUTE: the FREE 'compute' partition only - one shared always-on node with 8x NVIDIA L40 (48GB VRAM each). Your concurrent cap across all your jobs is 6 GPUs, 124 CPUs, and 384GB RAM; max wall time is 24h per job. Request 1 GPU (--gres=gpu:1) unless the experiment genuinely needs more, and up to 6 when it does (single node, so torchrun/FSDP works). Sizing on 48GB: fp16 inference up to ~20B params, LoRA/QLoRA fine-tuning up to ~7B (13B with care), full fine-tuning only up to ~2-3B. Add --qos=debug for validation runs under 2h to jump the queue. PAID elastic-* partitions (A100/H100) are NOT authorized and the account is not enabled for them: if an experiment needs more than 6 L40s or more VRAM per device, do not attempt it - record a FAIL-on-affordability with the exact resource ask and continue. WORKFLOW for each experiment: (1) write the code and an sbatch script locally under experiments/exp-NNN/; (2) stage it with rsync -avP experiments/exp-NNN/ mats:/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN/; (3) submit with ssh mats 'cd /mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN && mkdir -p logs && sbatch run.sbatch' - create logs/ BEFORE sbatch, because Slurm opens the output file at job launch and an in-script mkdir is too late; (4) poll with ssh mats 'squeue -u t.buckworth' until the job leaves the queue (PD pending, R running, CG completing), sleeping 60s between polls rather than busy-looping - the free partition is shared and a PD (Resources) wait of tens of minutes is normal, NOT a failure, so keep waiting; do NOT end the step while a job you submitted is still queued or running - if you must stop, first record the job id and its experiment in state.md so the next attempt resumes waiting instead of resubmitting; and before submitting anything, run squeue and check whether a job for this experiment is already queued or running, because the step may be retried after an interruption and duplicate submissions waste the shared partition; (5) pull results back with rsync -avP mats:.../exp-NNN/ experiments/exp-NNN/ and copy the full slurm log into run.log. NEVER run training, fine-tuning, heavy inference, or long CPU loops in an ssh shell on the dev node - it is the shared login node and that is the cluster's worst etiquette violation. Never run agents or jobs on the controller. The local machine this run executes on is for orchestration, plotting, and light analysis of returned results only - do not silently fall back to a local GPU for experiments under this profile. Job script requirements: all #SBATCH lines must precede the first command (any command above them silently disables every directive below); include --partition=compute, --gres=gpu:1, a realistic --time, --job-name, --cpus-per-task=8, --mem=32G, --output=logs/slurm-%j.out. Storage on the cluster: code, checkpoints, and final results under /mnt/nw/home/t.buckworth (persistent NFS, NOT backed up - pull anything irreplaceable back to this machine); HuggingFace cache, dataset shards, and intermediate outputs under /ephemeral/t.buckworth (fast local scratch, wiped on reboot) via HF_HOME=/ephemeral/t.buckworth/hf. Inside every job script: source ~/venv/bin/activate (the cluster's shared venv) and run python with -u so logs are not buffered. That venv pins torch 2.5.1+cu121 to match the workers' CUDA 12.2 driver - do NOT upgrade torch or install one from default PyPI, which breaks CUDA on every job. Diagnostics: ssh mats 'scontrol show job <jobid>' explains a stuck job; sacct works only from the controller (it errors with Connection refused on the dev node); nvidia-smi on the dev node fails because there is no GPU there, which is expected; on compute, nvidia-smi inside a job lists all 8 physical GPUs but you only own $CUDA_VISIBLE_DEVICES; an empty .out file on a running job is usually stdout buffering. scancel anything left idle. Prefer lightweight experiments (small open-weight models), each targeting under 30 min of GPU time. Max 5 experiments."
is_followup: false
novelty_verdict: NOVEL
criteria_approved: true
challenge_outcome: proceed_with_revisions
mentor_verdict: MAJOR_REVISIONS
clarifications:
  - q: "What is the motivating question (in the topic's own framing)?"
    a: "Can the existing Spectral Optimizer (a gradient-filtering wrapper around Adam/AdamW) be used to reduce the effect of noise when training models on large, noisy financial time-series data — i.e., does it improve out-of-sample (test-set) predictive performance relative to plain Adam/AdamW? The deliverable must return a verdict on whether the Spectral Optimizer HELPS on this data regime, not merely measure differences. Secondary question: is any benefit consistent across architectures — a plain MLP and at least one recurrent/state-of-the-art time-series architecture?"
  - q: "What do key terms mean?"
    a: "'Spectral Optimizer' = the EXISTING implementation from the prior project at /home/titus/pyg/optimizers (GitHub tbuckworth; do NOT recreate it — reuse the code). Two reusable variants exist: (1) experiments/spectral_optimizer.py — SpectralConsensusFilter, which computes per-sample gradients via torch.func vmap, eigendecomposes the B x B inter-sample gradient similarity matrix, and keeps only gradient directions many samples agree on (Marchenko-Pastur-style threshold, hard/soft/variance modes), wrapping any base optimizer; (2) experiments/weight_cov_optimizer_v2.py — a streaming rank-k gradient-covariance filter (~2x Adam cost, no B x B or p x p matrix formed). Prior findings (README.md): strongest result is label-noise robustness (MNIST @ 90% label noise: ~80% accuracy vs ~37% for Adam); mechanism characterized as a 'coherence amplifier' — helps when useful signal is the gradient-coherent component, hurts when signal is weak/not yet dominant (sparse parity). 'Noise reduction on financial time series' = financial return prediction is a low signal-to-noise regime; the hypothesis is that consensus/covariance filtering suppresses fitting of idiosyncratic noise and improves generalization. 'Stadia models / RushCursive models' in the issue is speech-to-text garbling, interpreted as: state-of-the-art / state-space time-series architectures, like recursive (recurrent) models — i.e., compare on a plain MLP AND on whatever architectures are standard for this task (e.g., LSTM/GRU, or a small state-space/Mamba-style or transformer model); the literature step should confirm what is standard for noisy financial tabular/time-series prediction."
  - q: "What is the implanted/target construct, and is it a faithful goal (not a fixed string)?"
    a: "N/A — this is not a covert/misaligned-behaviour topic. It is an empirical optimizer-evaluation study. The analogous validity concern: the evaluation must use genuinely noisy real (or realistically noisy) financial data with strict temporal train/test separation, so the outcome is not knowable a priori. The prior project's MNIST label-noise result does NOT predetermine the result here — financial noise is feature/target noise in a low-SNR regression regime, not synthetic label flips, and the prior work found the filter HURTS when signal is weak and not yet gradient-dominant, which is plausibly the financial regime. The outcome is genuinely uncertain in either direction."
  - q: "What prior work is known?"
    a: "The prior project itself: /home/titus/pyg/optimizers (README.md findings table H1-H7, research/spectral_filter_blog.html, research/literature_review.md). Related known lines: per-sample gradient agreement methods (e.g., gradient sign agreement / Agree-to-Disagree-style filtering, coherent gradients), random matrix theory / Marchenko-Pastur denoising of covariance matrices (well-established in quantitative finance for portfolio covariance, but there applied to RETURNS covariance, not per-sample GRADIENT covariance during training), and noisy-label robust training. The issue mentions the idea of a public leaderboard for financial prediction — the Numerai tournament dataset is the canonical large, noisy, obfuscated financial dataset with era-based out-of-sample evaluation and public downloads; plain stock/crypto return prediction with temporal splits is the fallback. Literature review (Step 2) should discover the rest."
  - q: "What does success look like?"
    a: "A proof of concept that returns a verdict on the motivating question: either (a) the Spectral Optimizer measurably improves out-of-sample performance vs tuned Adam/AdamW on at least one large noisy financial time-series task (with matched compute/hyperparameter budget and multiple seeds), characterized across at least two architectures (MLP + one recurrent/SOTA-style model), or (b) a well-documented negative result explaining why the coherence-amplification mechanism does not transfer to the low-SNR financial regime. A leaderboard-style held-out evaluation (e.g., Numerai validation eras) is desirable but a strict temporal test split suffices. Both outcomes are equally valuable."
  - q: "What is the scope and compute profile?"
    a: "Experiments run within the run's compute profile: the MATS Slurm cluster free 'compute' partition (8x L40 48GB node), driven remotely over SSH, max 6 concurrent GPUs, jobs sized to under ~30 min GPU time each, max 5 experiments. Do not exceed it; components needing more become future work. The workloads here (MLPs and small recurrent models on tabular/time-series data, per-sample gradients via vmap at moderate batch sizes) fit comfortably on a single L40 per job. Prefer computationally lightweight experiments: small models, subsampled eras/assets where needed. No paid data sources — use freely downloadable datasets (e.g., Numerai public dataset, Yahoo Finance-style OHLCV, or standard public benchmarks). No live leaderboard submission is required (submission portals need accounts/keys); a held-out leaderboard-equivalent split is acceptable."
  - q: "What assumptions are being made?"
    a: "Standard ML assumptions, plus: (1) the existing Spectral Optimizer code is reusable outside its home repo as designed (it wraps any base optimizer; SpectralConsensusFilter assumes a per-sample loss signature — regression losses like MSE need a small loss_fn swap, and its step(inputs, targets) interface must be integrated into the new training loop); (2) per-sample gradient computation via torch.func vmap works for the chosen architectures (recurrent models may need care); (3) a public noisy financial dataset with a defensible out-of-sample protocol is freely downloadable; (4) noise in financial data is predominantly gradient-incoherent, so the consensus filter's mechanism applies. All will be challenged in Step 6."
decisions:
  - step: 1
    decision: "Located the existing Spectral Optimizer (per issue checklist): /home/titus/pyg/optimizers/experiments/spectral_optimizer.py (SpectralConsensusFilter) and weight_cov_optimizer_v2.py (streaming variant). Reuse, do not recreate."
  - step: 1
    decision: "Interpreted garbled issue terms 'Stadia/RushCursive models' as state-of-the-art / recurrent (and possibly state-space) time-series architectures; literature review to confirm the standard comparison set."
  - step: 1
    decision: "Leaderboard requirement relaxed to a leaderboard-equivalent held-out evaluation (e.g., Numerai validation eras or strict temporal test split); no live submission required."
  - step: 2
    decision: "Literature review complete: 28 curated references. Closest prior work is Gradient Agreement Filtering (Chaubard et al. 2024, simple agreement checks, no RMT thresholding); no published work combines per-sample gradient MP-spectral filtering with financial prediction. Evaluation protocol must follow Numerai practitioner standards (era-purged CV, per-era correlation, FNC) and DeepOBS-style matched tuning budgets. Feldman's long-tail memorization theory identified as the sharpest counter-hypothesis."
  - step: 2
    decision: "Architecture note discovered: Numerai asset IDs reset each era, so a longitudinal recurrent arm cannot run on Numerai directly — the sequence-architecture arm needs within-era framing or an OHLCV-style dataset. Flagged for Steps 4-5 design."
  - step: 3
    decision: "Novelty verdict: NOVEL. Proceeding. Closest neighbor is Gradient Agreement Filtering (Chaubard et al. 2024) — same intervention level but pairwise orthogonality checks on image classification, not MP-spectral thresholding on financial regression. The combination of per-sample gradient covariance eigenspectrum filtering + low-SNR financial evaluation is unoccupied across all 28 references from three search channels. Caveats: novelty rests on absence of evidence in an active niche (date the search, position against GAF, don't claim a vacuum); the contribution is the transfer verdict, not the optimizer itself (author's own unpublished prior work)."
  - step: 4
    decision: "Success criteria defined and auto-approved (autonomous mode). Task-side SOTA treated as Numerai practitioner sanity ranges (mean per-era corr ~0.01-0.04, corr-Sharpe ~0.5-1.0; far-above-band numbers signal leakage, not success). Comparison-side bar is methodological: tuned AdamW at DeepOBS-style matched tuning budget, >=3 seeds, filter-on vs filter-off at identical base config as the cleanest mechanism isolation. Verdicts have teeth: 'helps'/'hurts' require paired per-era CI excluding zero plus practical significance (>= ~0.005 mean per-era corr); 'doesn't help' is an equivalence-style claim (CI excludes >=0.005 improvement), not p>0.05. Every verdict requires spectral engagement diagnostics (eigendirections kept vs MP bulk, gradient-norm fraction passed) so a null reads 'mechanism engaged but didn't help', not 'implementation no-op'. Venues: TMLR primary, ICAIF secondary, NeurIPS OPT workshop fallback. Minimum viable contribution (Numerai + MLP, 2 arms + ablation) fits in <=3 of 5 experiment slots. Designated cuts under budget pressure: Muon and GBT baselines — never seeds or matched tuning."
  - step: 5
    decision: "Steinhardt decomposition complete: 9 components, no SHOWSTOPPERs, no SKIPs. Highest-lambda component is spectral engagement on financial gradients (P=0.40, lambda=1.22) — the prior project's weak-signal failure mode and Feldman's long-tail theory both predict possible degeneracy in the low-SNR regime; a fail pivots to a reportable mechanistic finding, not project death. Pre-experiment gate (~4-6h wall-clock, two --qos=debug GPU jobs ~25 min combined) spends zero of the 5 experiment slots. Overall P: ~0.07 for the full clean verdict, ~0.30 for an informative publishable outcome (since #1-fail and #3-fail convert to countable deliverables); sequence arm (P=0.35 on vmap feasibility) is a stretch goal, not plan of record."
  - step: 6
    decision: "Three independent challenge passes complete (assumption-challenger, mentor-review, pre-mortem). Construct-validity gate PASSED: mentor-review explicitly confirms the construct is sound (outcome genuinely uncertain in both directions, faithful real-data evaluation, plan commits to a directional verdict) — no rethink, no loop to Step 1. Mentor verdict: MAJOR_REVISIONS, all revisions protocol-level and cheap; folded into the plan as 13 binding fix-now-free amendments + 5 fix-now-cheap additions in challenge/limitation-triage.md (no decomposition re-run needed — lambda ordering unchanged). Load-bearing fix: tuning/verdict era separation (F1). Other key amendments: pre-registered threshold-mode rule (F2), relative practical-significance threshold min(0.005, 25% of realized baseline) (F3), joint P(verdict reachable)>=0.6 go/no-go gate (F4), block-bootstrap inference with horizon-sized embargo (F5), cut order inverted so mechanism-discriminating controls outrank the sequence arm (F7), spiked-covariance + permuted-target diagnostics calibration (C1/C2), update-cosine degeneracy detector with pivot rule (C3). Steps 7 and 9 MUST read challenge/limitation-triage.md. Proceeding to Step 7."
  - step: 7
    decision: "Experiment plan auto-approved (autonomous mode). The 9 decomposition components exceed the 5-experiment cap, so they are packed into 5 experiments along the decomposition's own dependency/wave structure rather than cut top-5-by-lambda (a naive top-5 cut would keep #1 and #3 while severing their prerequisites #6/#9/#7 and #8, making the plan unexecutable). Packing: exp-001 = gate bundle #6+#9+#7+#1 (lambda driver 1.22); exp-002 = #2 (1.05, local CPU, scope decision not fail-fast); exp-003 = #8+#3+#4 with the F4 go/no-go gate; exp-004 = the seeded main comparison + C4 mechanism controls (the verdict deliverable); exp-005 = conditional sequence/second-setting arm (#5 fallback inside it). All F1-F13 amendments and C1-C5 additions from challenge/limitation-triage.md folded in as binding pass/fail modifications. Fail-fast semantics annotated per experiment: exp-001 #1-degeneracy and exp-003 leakage are genuine fail-fast triggers with pre-authorized pivots; exp-002 failure is a pre-authorized scope downgrade (continue MLP-only); exp-005 is conditional and cuttable per F7. No [SKIP] components existed to remove."
  - step: 8
    decision: "Fail-fast agreement recorded autonomously: CONDITIONAL. Blanket fail-fast (any FAIL terminates) would contradict the approved plan, in which exp-002/exp-005 failures are scope decisions and exp-004 no-verdict is a priced write-up outcome. Agreed protocol: execute in lambda order; on the two hard triggers (exp-001 mechanism degeneracy, exp-003 leakage) stop downstream work and take the pre-authorized pivot (mechanistic-finding deliverable / fix-split-first); the pre-mortem pivot indicators are adopted verbatim as kill criteria, most already binding via amendments F4 (P(verdict reachable)>=0.6 gate), C3 (degeneracy cosine + pivot rule), F5 (block-bootstrap collapse rule), F12 (hurts-verdict downgrade), F7 (never cut the last mechanism-discriminating control). No experiment is run past a tripped kill criterion."
  - step: 9
    decision: "exp-001 middle-case ruling (recorded for the Step 10 audit): headline #1 logged FAIL because the pre-registered PASS criterion required C2 real-vs-permuted distinguishability, which failed BY PROOF (for scalar-output MSE the engagement eigenspectrum is exactly target-independent). But the BINDING hard fail-fast trigger (all-modes degeneracy OR C3>0.95 everywhere) did NOT fire — the filter does sustained selective non-trivial work. Motivating question remained open and answerable; proceeded to exp-004 under an AMENDED INTERPRETATION FRAME: the target-independence proof is a co-primary deliverable; any verdict is attributed to target-blind spectral-subspace regularization, never 'consensus on signal'; C4 random-subspace control is load-bearing and protected. All pre-registered verdict machinery unchanged."
  - step: 9
    decision: "FINAL VERDICT (exp-004, pre-registered machinery): category HURTS — filter_on − filter_off = −0.00527 mean per-era numerai_corr, block-bootstrap 95% CI [−0.00886, −0.00181], exceeding F3=0.00398; corr-Sharpe −0.200 [−0.416, −0.004]. Mandatory F12 downgrade applies (spectral arm afforded ~1 tuning trial vs AdamW's 12): reportable claim = 'no evidence of benefit under the affordable tuning budget', raw hurts numbers reported alongside. Mechanism attribution: filter_on vs C4 norm/k-matched random subspace = −0.00116 CI [−0.00366, +0.00134] — MP-eigenselection indistinguishable from random subspace projection; all three filtered arms (spectral, random, GAF) hurt similarly; no Feldman dispersion-tail pattern; filter AMPLIFIES updates (ratio_med ~10x), not shrinkage."
  - step: 9
    decision: "exp-005 run (both authorization conditions held: exp-002 PASS, exp-003 budget room). Architecture consistency CONFIRMED on the GRU sequence arm, same data/machinery: headline −0.00484 CI [−0.00758, −0.00211] (vs MLP −0.00527), same HURTS→F12-downgrade category; spectral-vs-random null replicates (−0.00011 CI [−0.00233, +0.00215]); corr-Sharpe −0.242 [−0.414, −0.073]. F13 satisfied via within-era Numerai framing (T=15 x D=47 reshape) — 'architecture consistency' claim supported, OHLCV fallback unused. Notable: filter engages in a different regime on the GRU (k~60, mild attenuation) vs MLP (k~1, ~10x amplification) yet produces near-identical harm and the same null — strengthening the 'generic subspace projection/rescaling, not spectral selection' account. Limitation recorded: GRU used t07 fallback config (tuning-shard presence misjudged from login node — /ephemeral is node-local); config was frozen pre-unblinding and identical across arms, so the paired comparison is protocol-valid; ~10 GPU-min resource ask carried to Future Work."
  - step: 10
    decision: "Round-1 results audit complete (fresh results-auditor agent; audit/results-audit.md). Overall disposition: HONEST-NEGATIVE, audit_exit_reason: true-null — NO remediation round needed, advance to Step 11. The load-bearing exp-004 was RE-EXECUTED on the cluster with fresh seeds {10,11,12} and an auditor-built eval shard (independent row subsample from raw v5.0_validation.parquet; jobs 6616/6618/6620, ~9 min L40): headline REPRODUCES in category/direction/magnitude (producer shard -0.00546 CI [-0.00892,-0.00204]; audit shard -0.00562 CI [-0.00911,-0.00201]; 6-seed pool -0.00537 CI [-0.00853,-0.00222]); C4 spectral-vs-random null replicates; target-independence proof independently re-verified (2.1e-14, different construction) with two new scoping controls showing the theorem is non-vacuous. 8 findings: 5 SUPPORTED, 2 FIXABLE-DEFECT both write-up-level and non-verdict-flipping (NEW Finding 5: producer code crashes stochastically by seed via linalg.eigh non-convergence under the documented rank-collapse regime — original results are 6-of-6-seed lucky-complete, disclosure required; Finding 6: MLP corr-Sharpe CI upper bound is bootstrap-RNG fragile, word as marginal and let the robust GRU Sharpe carry the stability claim), 1 TRUE-NULL (Finding 7: verdict-block baseline +0.0064 below the F9 sanity band — regime drift, claim must be scoped to this low-edge regime; F3 realized at ~62% relative, not 25%). No gaming signatures, no criteria drift (exp-001 middle-case ruling examined and cleared as conservative/pre-unblinding/claim-narrowing — paper must keep exp-001 headline as logged FAIL-on-C2). Step 11 MUST use the audit's 8 unresolved-for-write-up items and the 15-row limitation triage (6 fix-now-free, 3 fix-now-cheap, 6 future-work with resource asks)."
lambda_table:
  - component: "Spectral engagement on financial gradients"
    p_success: 0.40
    t_hours: 0.75
    lambda: 1.22
    status: "FAIL on C2 conjunct only (target-independence PROOF; banked mechanistic finding — hard fail-fast trigger did NOT fire; selectivity + C3 clean)"
  - component: "Sequence-arm vmap feasibility (recurrent model)"
    p_success: 0.35
    t_hours: 1.0
    lambda: 1.05
    status: PASS
  - component: "Statistical power of paired per-era design"
    p_success: 0.60
    t_hours: 0.5
    lambda: 1.02
    status: PASS
  - component: "Full design fits 5-experiment / 30-min budget"
    p_success: 0.80
    t_hours: 0.25
    lambda: 0.89
    status: PASS
  - component: "OHLCV fallback dataset for sequence arm"
    p_success: 0.80
    t_hours: 0.5
    lambda: 0.45
    status: "NOT NEEDED (within-era Numerai framing worked for exp-005)"
  - component: "Spectral Optimizer regression integration"
    p_success: 0.75
    t_hours: 0.75
    lambda: 0.38
    status: PASS
  - component: "vmap throughput on L40 within job cap"
    p_success: 0.85
    t_hours: 0.5
    lambda: 0.33
    status: PASS
  - component: "Baseline sanity: tuned AdamW in sane corr band"
    p_success: 0.70
    t_hours: 1.5
    lambda: 0.24
    status: PASS
  - component: "Numerai data + era-purged protocol"
    p_success: 0.85
    t_hours: 1.0
    lambda: 0.16
    status: PASS
approved_experiments:
  - id: exp-001
    name: "Gate bundle: integration, data/protocol, throughput, spectral engagement"
    components: "#6 (lambda 0.38), #9 (0.16), #7 (0.33), #1 (1.22 — lambda driver)"
    compute: "Local CPU (#6, #9) + one shared --qos=debug GPU job (#7 + #1), ~15-20 min GPU"
    amendments: "F1 (tuning/verdict era split built at #9), F2 (threshold mode fixed from engagement diagnostics only, before any OOS performance seen), F5 (embargo >= ceil(horizon/era-spacing)), F6 (diagnostics under BOTH within-era and mixed-era batch compositions), F9 (sanity/leakage bands recalibrated from current Numerai v5 example-model stats), F11 (optimizer identity = spectral_optimizer.py exact B x B variant; streaming swap = scope change, not silent fallback), C1 (spiked-covariance unit test in #6 smoke: iid noise -> keeps ~0; planted spike -> kept; correlated zero-signal confound -> measured), C2 (permuted-target null-spectrum control in the debug job), C3 (filtered-vs-mean-gradient and filtered-vs-unfiltered cosine diagnostics), C5 (era-identity probe of the kept subspace)"
    pass: "#6: loss falls on synthetic regression, both variants run, filter-off reproduces plain AdamW to tolerance, C1 all three cases behave correctly. #9: Numerai v5 downloaded, purged split with F1 tuning/verdict block separation and F5 embargo, enough usable eras in BOTH blocks (target >=100 total validation-side), <32GB, shard readable on cluster. #7: projected full training run <20 min at the batch size #1 needs. #1: for at least one threshold mode, sustained selective filtering (0 < eigendirections kept < B; grad-norm fraction passed in ~[0.1, 0.9]) at B in {256, 1024} under at least one batch composition, AND C3 mean-gradient cosine <= ~0.95 (not pure mean-gradient smoothing), with C2 permuted-target spectra distinguishable from real-target spectra."
    fail: "#1: all modes degenerate (~0% or ~100% norm passed) at both batch sizes and both compositions, OR C3 cosine > 0.95 everywhere (filter ~= mean-gradient smoothing degeneracy)."
    on_fail: "FAIL-FAST TRIGGER with pre-authorized pivot: the helps/hurts comparison is uninterpretable; deliverable pivots to the mechanistic finding ('no gradient-coherent signal separable from the MP bulk on Numerai at feasible batch sizes' or 'filter degenerates to mean-gradient smoothing'), characterized against the C2 permuted-target null and the prior MNIST-label-noise spectra. Remaining budget shifts to characterizing why. Infrastructure sub-failures (#6/#9/#7) are fixed in place (code is ours and small); #7-forced variant swap invokes F11 scope-change rules."
  - id: exp-002
    name: "Sequence-arm vmap feasibility (hand-rolled GRU cell)"
    components: "#2 (lambda 1.05)"
    compute: "Local CPU only (correctness smoke); optional 5-min --qos=debug job for throughput"
    amendments: "F13 (claim naming depends on dataset held fixed)"
    pass: "vmap(grad) per-sample gradients through a functional GRU cell match a per-sample loop reference to ~1e-5 on within-era sequences, AND projected sequence-model training run <25 min on one L40."
    fail: "vmap errors requiring >2h of surgery, or projected runtime over the job cap."
    on_fail: "NOT fail-fast — pre-authorized scope decision: drop to MLP-only minimum viable contribution, remove the architecture-consistency claim explicitly, free the exp-005 slot per F7 cut order (goes to mechanism controls/tail-era analysis). Continue to exp-003 regardless of outcome."
  - id: exp-003
    name: "Baseline sanity, power analysis, budget fit, and F4 go/no-go gate"
    components: "#8 (lambda 0.24), #3 (1.02), #4 (0.89)"
    compute: "One GPU array-style job (8-12 AdamW tuning trials, ~1-2 min each on subsampled eras) + local CPU bootstrap and arithmetic"
    amendments: "F1 (sweep touches TUNING-block eras only; power computed on VERDICT-block era count), F3 (register verdict threshold = min(0.005, 0.25 x realized tuned-baseline mean per-era corr) before unblinding any comparison), F4 (proceed to exp-004 only if moving-block bootstrap gives P(some verdict category reachable) >= 0.6; else re-scope the endpoint BEFORE spending slots), F5 (check lag-1 autocorrelation of per-era corr; all CIs and the power sim use moving-block bootstrap with block length >= target-horizon overlap), F9 (recalibrated corr sanity band and leakage gate), F12 (spectral-arm priors transferred from ~/pyg/optimizers as search center; log grid-boundary / non-monotone under-exploration signatures), F7 (budget-fit arithmetic uses the inverted conditional cut order)"
    pass: "#8: best trial's mean per-era corr inside the F9-recalibrated sane band on tuning-block eras, zero-predictor |corr| < 0.002, no leakage signature. #3: block-bootstrap CI half-width <= ~F3-threshold achievable at 3 seeds x verdict-block eras. #4: full plan (tuning both arms + 2 arms x 3 seeds + C4 controls + diagnostics [+ sequence arm]) packs into remaining slots at <=25 min/job with the F7 cut order. F4 gate: P(verdict reachable) >= 0.6."
    fail: "All trials ~0 corr (< 0.003), OR any trial far above band (leakage signature), OR CI half-width > ~0.008 using all verdict-block eras, OR minimum viable design cannot fit even after identical-across-arms subsampling."
    on_fail: "~0 corr: reduce subsampling / switch to a corr-aligned loss (held fixed across arms), retry once. LEAKAGE: FAIL-FAST TRIGGER — fix the purge/embargo before anything else runs; nothing downstream is interpretable until fixed. Power fail: first response free (more verdict-block eras); else re-scope per F4 to an honestly-scoped effect estimate with calibrated uncertainty instead of a categorical verdict. Budget fail: execute F7 cut order (Muon -> GBT -> sequence arm -> GAF -> random-subspace control; never seeds, never matched tuning, never verdict-block separation); if minimum viable still doesn't fit, record FAIL-on-affordability with the exact resource ask."
  - id: exp-004
    name: "Seeded main comparison + mechanism controls (the verdict experiment)"
    components: "Main deliverable (builds on #1, #8, #3; success-criteria verdict definitions as amended)"
    compute: "GPU array-style job(s): 2 arms (spectral filter-on vs filter-off tuned AdamW, identical base config) x 3 seeds on subsampled Numerai; C4 controls co-scheduled in the same arrays where possible"
    amendments: "F1 (evaluated ONCE on verdict-block eras; no peeking before arms are frozen), F2 (threshold mode is whatever exp-001 pre-registered), F3 (practical-significance threshold as registered in exp-003), F5 (paired per-era moving-block bootstrap CIs), F7 (mechanism-discriminating controls protected ahead of optional arms), F8 (tail-era / era-quantile breakdown is a mandatory analysis output, promoted to minimum-viable deliverable), F10 (corr-Sharpe route to a verdict tied to the same paired block-bootstrap CI machinery or dropped), F12 (if verdict is 'hurts' AND under-exploration signatures present, downgrade to 'no evidence of benefit under the affordable tuning budget'), C4 (random-subspace norm-matched control at the measured kept-norm fraction; GAF-style simple-agreement ablation if it co-schedules)"
    pass: "A three-way verdict (helps / hurts / doesn't help) is returned per the amended success criteria: paired per-era block-bootstrap CI + F3 practical significance for helps/hurts; equivalence-style CI exclusion for doesn't-help; spectral engagement diagnostics logged throughout so the verdict is attributable to an engaged mechanism; C4 control separates MP-selection from generic update-norm shrinkage; F8 tail-era breakdown reported."
    fail: "No verdict category reachable after the run (CI too wide despite the F4 gate), or diagnostics show the mechanism silently disengaged during the verdict runs."
    on_fail: "Report the effect estimate with calibrated uncertainty, honestly scoped to the CI achieved ('could not determine' is the priced residual-risk outcome, ~1-in-4 to 1-in-3); mechanism-disengagement mid-run reopens the exp-001 mechanistic-finding pivot. Either way this is a write-up path, not a retry loop."
  - id: exp-005
    name: "Sequence/second-setting arm (conditional stretch goal)"
    components: "#2 outcome + #5 (lambda 0.45) as internal fallback"
    compute: "One GPU job <25 min (small sequence model); #5 dataset build is local"
    amendments: "F7 (this arm is cut BEFORE the GAF ablation and random-subspace control under budget pressure), F13 (if run on OHLCV rather than within-era Numerai, either add a cheap MLP-on-OHLCV arm or claim 'second setting', NOT 'architecture consistency')"
    pass: "Sequence model trains within cap on within-era Numerai framing (preferred) or the #5 OHLCV protocol (>=90% coverage after cleaning, leak-free temporal split with purge+embargo, trivial baseline evaluates cleanly); paired comparison of filter-on vs filter-off with the same inference machinery as exp-004."
    fail: "Within-era framing fails AND #5 fails (>2h cleaning or no defensible protocol), or the arm doesn't fit the remaining budget."
    on_fail: "NOT fail-fast — cut per F7, note the removed claim explicitly as a limitation, carry to Future Work as W2 with its resource ask. Run only if exp-002 passed AND exp-003's #4 arithmetic left room."
fail_fast_agreement: conditional
fail_fast_conditions: "Hard fail-fast triggers (stop/pivot per pre-authorized on_fail): exp-001 #1-degeneracy (all threshold modes ~0%/~100% norm passed at both batch sizes and compositions, OR C3 mean-gradient cosine > 0.95 everywhere) and exp-003 leakage signature (fix purge/embargo before anything downstream runs). NOT fail-fast: exp-002 and exp-005 failures are pre-authorized scope decisions (MLP-only / cut per F7) and never stop the run; exp-004 no-verdict is a write-up path (honest effect estimate), never a retry loop. Kill criteria: the five pre-mortem pivot indicators recorded in the Step 8 narrative are binding stopping/pivot conditions for Step 9."
experiments_completed: ["exp-001 (infra PASS; headline #1 FAIL on C2 conjunct only — banked mechanistic finding, no hard trigger)", "exp-002 (PASS)", "exp-003 (PASS, F4 gate GO)", "exp-004 (PASS — verdict returned)", "exp-005 (PASS — architecture consistency confirmed)"]
experiments_failed: []
final_verdict_category: "hurts (pre-registered category) -> F12 downgrade applied"
final_verdict_wording: "No evidence of benefit under the affordable tuning budget; at the pre-registered operating point the Spectral Optimizer significantly degraded out-of-sample performance (MLP: -0.00527 CI [-0.00886, -0.00181]; GRU: -0.00484 CI [-0.00758, -0.00211] mean per-era numerai_corr), and MP-eigenselection was indistinguishable from a norm/k-matched random-subspace control on both architectures."
audit_round: 1
audit_exit_reason: "true-null"
---

# Workflow Progress

## Step 1: Clarifications

Step 1 complete. Topic clarified autonomously.

Topic: does the existing Spectral Optimizer (gradient consensus/covariance
filtering from ~/pyg/optimizers) reduce the effect of noise when training on
large, noisy financial time-series data — i.e., does it beat Adam/AdamW
out-of-sample, and is the effect consistent across an MLP and a
recurrent/SOTA-style time-series architecture?

Key Step 1 findings:
- The existing optimizer was located (issue checklist item): two reusable
  implementations in /home/titus/pyg/optimizers/experiments/
  (spectral_optimizer.py = exact per-batch consensus filter;
  weight_cov_optimizer_v2.py = streaming rank-k covariance filter, ~2x Adam).
- Prior project's strongest result is label-noise robustness (MNIST @ 90%
  noise), but it also found the filter hurts when signal is weak and not
  gradient-dominant — so transfer to the low-SNR financial regime is genuinely
  uncertain in both directions.
- Candidate dataset: Numerai public tournament data (large, noisy, obfuscated
  financial features, era-based out-of-sample evaluation — closest free thing
  to the issue's 'leaderboard' ask); fallback is public OHLCV return
  prediction with strict temporal splits.

## Step 2: Literature Review

Complete. Search plan (`search-plan.md`) executed via three parallel search
agents; 28 curated references in `references.bib` (22 academic, 2 lab-blog,
4 community), synthesized in `literature/synthesis.md`.

Key findings:
- **Closest prior work**: Gradient Agreement Filtering (Chaubard et al. 2024)
  — cross-microbatch agreement filtering, but simple orthogonality checks,
  no RMT/Marchenko-Pastur spectral thresholding, and not on financial data.
  A 2025-2026 burst of per-sample-gradient optimizer work (OrthoGrad,
  PS-Clip-SGD, GradSentry, DP-PMLF) confirms the niche is active, but none
  do covariance-eigenspectrum filtering as the update rule.
- **Theoretical ancestor**: Coherent Gradients (Chatterjee & Zielinski 2020);
  independently articulated informally on the Alignment Forum (Pope 2022).
- **RMT analogy base**: Marchenko-Pastur covariance cleaning is a mature
  ~20-year quant-finance literature (Bouchaud/Potters et al.) — applied to
  RETURN covariance, never to training-gradient covariance. The transposition
  appears unpublished.
- **The bar is high**: Adam-family optimizers already have heavy-tailed-noise
  robustness (sign/normalization theory), and a 2026 tabular-optimizer
  benchmark finds Muon beats AdamW on MLP/tabular tasks. Fair comparison
  requires DeepOBS-style matched tuning budgets and multiple seeds.
- **Evaluation protocol** (from Numerai practitioner sources): era-purged
  temporal CV, per-era correlation as metric/loss, Feature Neutral
  Correlation to distinguish real signal from feature overfitting.
- **Sharpest counter-hypothesis**: Feldman 2019 — memorization of rare
  (long-tail) examples can be NECESSARY for generalization; financial data
  is heavy-tailed, so consensus filtering could suppress rare genuine signal.
  Aligns with the prior project's weak-signal failure mode. Outcome genuinely
  uncertain in both directions.
- **Design constraint discovered**: Numerai asset IDs reset each era — the
  recurrent/sequence arm needs within-era framing or an OHLCV-style dataset.
- **Confirmed novelty gap**: no academic Numerai literature; no work combining
  per-sample gradient RMT filtering with financial prediction, across three
  independent search channels.

## Step 3: Novelty Assessment

Verdict: **NOVEL** (with caveats). Full assessment in `novelty-assessment.md`.

- The combination of (a) per-sample gradient covariance eigendecomposition
  with a Marchenko-Pastur threshold as the training update rule and (b)
  evaluation on low-SNR financial prediction is unoccupied across all 28
  references from three independent search channels. The two adjacent
  literatures each stop short: quant-finance RMT cleans RETURNS covariance
  post-hoc, never training gradients; the 2024-2026 per-sample-gradient
  optimizer burst uses pairwise/geometric/clipping heuristics, never full
  covariance-eigenspectrum thresholding, and never on financial data.
- Closest neighbor: Gradient Agreement Filtering (Chaubard et al. 2024) —
  classified Related, not Near-identical.
- The domain transfer is non-trivial: both the prior project's weak-signal
  failure mode and Feldman's long-tail theory predict the mechanism could
  hurt, so the outcome is genuinely uncertain and informative either way.
- Caveats: (1) novelty rests on absence of evidence in an actively-producing
  niche — date the search and position against GAF rather than claim a
  vacuum; (2) the contribution is the transfer verdict, not the optimizer
  itself.
- Recommendations for downstream steps: optional GAF-style simple-agreement
  ablation arm; tuned AdamW + Muon-if-budget baselines under DeepOBS-style
  matched tuning; Numerai era-purged/FNC protocol; separate reporting on
  tail eras to test the Feldman counter-hypothesis directly.

## Step 4: Success Criteria

Defined and auto-approved (autonomous mode). Full criteria in
`success-criteria.md`.

- **SOTA framing split in two.** Task side: Numerai practitioner ranges
  (mean per-era corr ~0.01-0.04, corr-Sharpe ~0.5-1.0, FNC smaller) used as
  sanity ranges, not targets — numbers far above the band signal leakage.
  Comparison side: the bar is methodological (matched tuning budgets, >=3
  seeds, tuned AdamW; default-Adam-only would be a strawman given the 2026
  tabular benchmark showing Muon beats AdamW). There is no direct SOTA for
  the specific question — that is what makes any competent answer informative.
- **Required baselines** (by load-bearing weight): (1) tuned AdamW at matched
  budget; (2) filter-on vs filter-off at identical base config — the cleanest
  isolation of the mechanism; (3) GAF-style simple-agreement ablation (first
  designated cut); (4) Muon (secondary, cuttable with noted limitation);
  (5) zero-predictor sanity baseline.
- **Verdict definitions with teeth.** 'Helps'/'hurts' require a paired
  per-era CI excluding zero AND practical significance (>= ~0.005 mean
  per-era corr, ~20-25% relative). 'Doesn't help' is an equivalence-style
  statement (CI narrow enough to exclude >=0.005 improvements), not p>0.05.
  Every verdict requires spectral engagement diagnostics (eigendirections
  kept vs MP bulk, gradient-norm fraction passed) so a null means 'mechanism
  engaged but didn't generalize better', not 'implementation no-op'.
- **Publishability**: TMLR primary (values negative results, no
  mechanism-novelty bar — matches the 'contribution is the transfer verdict'
  framing), ICAIF secondary, NeurIPS OPT workshop fallback.
- **Budget fit**: minimum viable contribution (Numerai + MLP, 2 arms +
  ablation) fits in <=3 of 5 experiment slots, leaving >=2 for the sequence
  arm and GAF ablation. Cuts under pressure: Muon and GBT baselines — never
  the seeds or the matched tuning budget. Paired per-era design is the power
  mitigation: era count, not model size, drives statistical power and is
  nearly free within the <30-min job cap.

## Step 5: Steinhardt Decomposition

Complete. Full decomposition with component details, dependency graph, and
parallelisation plan in `decomposition.md`. Nine components across four
layers (infrastructure, cluster feasibility, scientific preconditions,
scope extension). No SHOWSTOPPERs (nothing below P=0.05, nothing exceeding
one L40 per job); no SKIPs (nothing >=0.9 with replications of our exact
usage).

Fail-fast ordering (top 3 by lambda):
1. **Spectral engagement on financial gradients** (P=0.40, lambda=1.22) —
   the scientific heart. Prior project's weak-signal failure mode + Feldman's
   long-tail theory both predict possible degeneracy (filter passes ~0% or
   ~100% of gradient norm) in the low-SNR regime. Tested with one short
   diagnostics-logging GPU run. A FAIL pivots to a reportable mechanistic
   finding ("no gradient-coherent signal separable from the MP bulk on
   Numerai"), not project death.
2. **Sequence-arm vmap feasibility** (P=0.35, lambda=1.05) — torch.func vmap
   doesn't compose with cuDNN RNNs; tested locally on CPU with a hand-rolled
   GRU cell. FAIL drops to MLP-only scope (pre-authorized fallback).
3. **Statistical power** (P=0.60, lambda=1.02) — whether the paired per-era
   CI can resolve the 0.005 verdict threshold at 3 seeds; tested by CPU
   bootstrap from the baseline's per-era corr vector. FAIL first mitigated
   free (more eras), else scope claim to the CI the design supports.

Structural notes:
- Pre-experiment gate (~4-6 h wall-clock) uses only two short --qos=debug
  validation jobs (~25 min GPU combined) and spends ZERO of the 5 experiment
  slots: #7 (vmap throughput) and #1 (spectral engagement) share one job;
  #8 (tuned-AdamW baseline sanity) is a parallel array job.
- Wave plan: [#6 integration smoke, #9 Numerai data] locally in parallel;
  then the two debug GPU jobs; then CPU-only power (#3) and budget-fit (#4)
  checks. #5 (OHLCV fallback) only if the Numerai within-era framing fails.
- Baseline sanity (#8) is the leakage gate: corr > 0.06 means the
  purge/embargo is broken and nothing downstream is interpretable until
  fixed.
- Overall P_success: ~0.07 for the full clean verdict, but ~0.30 for an
  informative publishable outcome (component #1 and #3 failures convert to
  explicitly countable deliverables under the success criteria). Uncertainty
  is concentrated in the science (#1, #3), not the infrastructure. The
  sequence-arm TMLR-tier extension (~0.03) is a stretch goal, not plan of
  record.

## Step 6: Challenge the Research Plan

Complete. Three independent adversarial passes dispatched in parallel, no
cross-reading: `challenge/assumption-analysis.md` (12 assumptions: 4
critical, 5 moderate, 3 background), `challenge/mentor-review.md` (verdict:
MAJOR_REVISIONS), `challenge/pre-mortem.md` (5 systemic failure scenarios).
Design-time limitation triage in `challenge/limitation-triage.md` — **Steps
7 and 9 must read it**; its fix-now amendments are binding on the experiment
plan.

**Construct-validity gate: PASSED.** The mentor review examined it directly:
the outcome is not statable on paper (prior evidence points both directions;
Feldman's theory gives a principled harm mechanism), the evaluation object is
faithful (real obfuscated financial data, era-purged temporal splits), and
the plan answers the motivating question with a three-way directional
verdict. No construct redesign; no loop to Step 1.

**Why MAJOR_REVISIONS did not trigger a loop back to Step 5**: every required
revision is a protocol amendment (era splits, pre-registration rules,
inference machinery, diagnostics additions, cut-order changes) that leaves
the component set and lambda ordering intact. All fixes folded into
`challenge/limitation-triage.md` as binding amendments instead — autonomous
mode, loop budget conserved.

**Convergent findings across the three independent passes** (highest
confidence):
1. **Selection/inference contamination** (mentor #1 = the load-bearing gap;
   pre-mortem #1/#3; assumption #7): no tuning/verdict era separation, an
   absolute 0.005 threshold that silently becomes a ~70% relative bar if the
   subsampled baseline lands low, and era-autocorrelated targets breaking
   naive bootstrap CIs. Fixes F1, F3, F4, F5 — all free.
2. **Diagnostics validate execution, not interpretation** (assumption #1/#3;
   mentor #3; pre-mortem #2): the MP threshold's null assumes i.i.d. samples,
   but financial per-sample gradients are cross-sectionally correlated
   (factor structure), so "engaged" can be spurious — and in low-SNR
   regression the spectrum may have only trivial rank-1 (mean-gradient)
   structure, making the filter a smoother that LR tuning absorbs. Fixes
   C1, C2, C3, C5 + F2, F6 — calibration tests and cosine diagnostics in
   the existing gate jobs.
3. **The ablation doesn't isolate the mechanism** (assumption #2; mentor's
   random-subspace suggestion; pre-mortem #2): filtering also shrinks update
   norm — implicit regularization known to help on noisy data. Fix C4:
   random-subspace norm-matched control, ranked above the GAF ablation.
4. **The cut cascade could strip a null of its meaning** (pre-mortem #5;
   mentor cut-order inconsistency; assumption rec 5): fix F7 — cut order
   inverted and made conditional; mechanism-discriminating controls outrank
   the sequence arm.
5. **Negative direction under-guarded** (pre-mortem cross-cutting theme):
   the rigor machinery targets false positives, but this project will
   readily publish a negative — fixes F12 (tuning-adequacy signatures,
   verdict downgrade rule) and C3 (degeneracy pivot rule) guard the
   false-negative direction.

**Accepted residual risk** (stated knowingly): ~1-in-4 to 1-in-3 chance the
deliverable is a caveated "could not determine" below the primary venue,
even with all mitigations — consistent with the decomposition's own 0.07
clean-verdict estimate, and acceptable because boundary/degeneracy findings
are explicitly priced as deliverables.

Outcome: `challenge_outcome: proceed_with_revisions`. Next: Step 7 (report
planned experiments), which must apply the F1-F13 amendments and the
C1-C5 additions when writing the approved experiment plan.

## Step 7: Approved Experiment Plan

Auto-approved (autonomous mode). Full machine-readable plan in the
`approved_experiments` frontmatter above; binding amendments from
`challenge/limitation-triage.md` (F1-F13, C1-C5) are folded into each
experiment's pass/fail criteria and Step 9 must honor them.

**Packing decision.** The decomposition's 9 components exceed the
5-experiment cap. A literal top-5-by-lambda cut would keep #1/#2/#3/#4/#5
while severing #1's prerequisites (#6, #9, #7) and #3's input (#8),
making the surviving experiments unexecutable. Instead the components are
packed into 5 experiments along the decomposition's own dependency waves —
each experiment is still lambda-led (the bundle runs its highest-lambda
question as the pass/fail headline) and the execution order preserves
fail-fast:

| Exp | Bundles | Lambda driver | Compute | Fail semantics |
|-----|---------|---------------|---------|----------------|
| exp-001 | #6 integration + #9 Numerai/protocol + #7 throughput + #1 spectral engagement | 1.22 | local CPU + 1 shared --qos=debug GPU job | #1-degeneracy = FAIL-FAST → mechanistic-finding pivot |
| exp-002 | #2 sequence-arm vmap feasibility | 1.05 | local CPU | scope decision only — MLP-only fallback, always continue |
| exp-003 | #8 baseline sanity + #3 power + #4 budget fit + F4 go/no-go | 1.02 | 1 GPU array job + CPU | leakage = FAIL-FAST (fix before anything runs); power/budget fails re-scope per F4/F7 |
| exp-004 | Seeded main comparison (2 arms x 3 seeds) + C4 mechanism controls | verdict deliverable | GPU array job(s) | no-verdict → honest effect estimate write-up, no retry |
| exp-005 | Sequence/second-setting arm (#5 OHLCV fallback internal) | 0.45 (stretch) | 1 GPU job, conditional | cut per F7, never blocks the run |

**Key amendment placements**: F1 era separation is built in exp-001 (#9)
and enforced everywhere downstream; F2 fixes the threshold mode at exp-001
from diagnostics only; F3's relative threshold and the F4 go/no-go gate are
registered in exp-003 before any comparison is unblinded; F5 block-bootstrap
inference governs exp-003's power sim and exp-004's verdict CIs; C1/C2/C3/C5
diagnostics-calibration all ride inside exp-001's existing smoke test and
debug job at ~zero marginal cost; C4's random-subspace norm-matched control
co-schedules with exp-004's arrays; F7's inverted cut order protects the
mechanism-discriminating controls ahead of the sequence arm; F8 makes the
tail-era breakdown a mandatory exp-004 analysis output; F11 pins the
optimizer identity (exact B x B SpectralConsensusFilter — a streaming-variant
swap is a declared scope change).

**Parallelism**: exp-001's local parts, exp-002, and exp-001's data download
can proceed concurrently; exp-001's debug GPU job and exp-003's array job
are submittable simultaneously (2 GPUs, within the 6-GPU cap). exp-004 waits
on the F4 gate; exp-005 waits on exp-002 + exp-003's budget arithmetic.

**No [SKIP] components existed** (nothing at P >= 0.9), so nothing was
removed; the cap is honored by bundling, and `rethink_disproof` is not set.

Step 7 complete. Next: Step 8 (fail-fast agreement).

## Step 8: Fail-Fast Agreement

Recorded autonomously: `fail_fast_agreement: conditional` — fail fast with
pre-authorized pivots, not blanket termination. This matches the fail
semantics already annotated per-experiment in the approved Step 7 plan;
a blanket `true` would wrongly terminate the run on exp-002 (a scope
decision) or exp-004 no-verdict (a priced write-up outcome).

**What fail-fast means here, per experiment:**
- **exp-001 (#1 spectral engagement, the lambda driver at 1.22)** — hard
  trigger. If all threshold modes degenerate (~0% or ~100% of gradient norm
  passed at both batch sizes and both batch compositions), or the C3
  filtered-vs-mean-gradient cosine exceeds ~0.95 everywhere, the helps/hurts
  comparison is uninterpretable: stop the comparison track and take the
  pre-authorized pivot (characterize the degeneracy against the C2
  permuted-target null and the prior MNIST label-noise spectra — a
  mechanistic finding, not project death).
- **exp-003 leakage signature** — hard trigger. Nothing downstream is
  interpretable until the purge/embargo is fixed; no further slots are spent
  before the fix.
- **exp-002 (sequence-arm vmap)** — NOT fail-fast. Failure drops scope to
  MLP-only (pre-authorized); the run always continues to exp-003.
- **exp-004 no-verdict** — NOT a retry trigger. The pre-registered fallback
  is an honestly-scoped effect estimate with calibrated uncertainty
  (the ~1-in-4 to 1-in-3 priced residual risk).
- **exp-005** — conditional stretch goal; cut per F7, never blocks the run.

**Kill criteria adopted from the pre-mortem** (its five pivot indicators,
binding on Step 9 — most already enforced as triage amendments):
1. **Endpoint-unreachable gate (F4)**: after exp-003's #8 and #3, if the
   joint block-bootstrap gives P(any verdict category reachable) < 0.6 under
   the realized baseline corr and era count, stop and re-scope the endpoint
   BEFORE spending the main-comparison slots.
2. **Degeneracy pivot (C3)**: update-cosine to the plain mean gradient
   > ~0.95 for all threshold modes across both tested batch sizes in the
   exp-001 gate run — stop treating the study as filter-vs-baseline;
   redesign around characterizing the degeneracy.
3. **Autocorrelation collapse (F5)**: if the moving-block bootstrap CI
   half-width exceeds the verdict threshold even at the full available era
   count, this collapses into kill criterion 1's endpoint redesign.
4. **Under-tuning downgrade (F12)**: a 'hurts' verdict accompanied by any
   under-exploration signature (grid-boundary best config, non-monotone
   sweep, inter-arm optimal-LR shift) is downgraded to 'no evidence of
   benefit under the affordable tuning budget' before write-up — a claim
   correction, not a stop.
5. **Contribution-identity floor (F7)**: if exp-003's budget arithmetic
   cannot fit the main comparison plus at least one mechanism-discriminating
   control (C4 random-subspace or GAF-style), re-scope the tuning sweep
   (fewer trials, declared) rather than cutting the last discriminating
   ablation.

Step 8 complete. Next: Step 9 (execute experiments), which must honor these
conditions, the per-experiment on_fail semantics, and the F1-F13 / C1-C5
amendments in challenge/limitation-triage.md.

## Step 9: Experiment Execution — COMPLETE

All five approved experiments executed. No hard fail-fast trigger fired; no
kill criterion tripped. Total GPU spend across the run: ~15 min on single
L40s (jobs 6430, 6432, 6439, 6614 — all --qos=debug, free compute
partition), far under budget.

- **exp-002 (sequence-arm vmap feasibility): PASS** — vmap(grad) per-sample
  gradients through a hand-rolled functional GRU match the loop reference to
  1.7e-06; full filtered step 1.17 s on CPU at B=1024 (eigh only ~3% of it);
  projected sequence-model training ~3.9 min on one L40 under a conservative
  10x scaling (budget met at any speedup >=1.6x); end-to-end
  SpectralConsensusFilter step on GRU gradients ran (k=4 kept,
  consensus_ratio 0.457). No MLP-only fallback needed; the sequence arm
  supports the "architecture consistency" claim route if it stays on
  within-era Numerai (F13). Caveat: verified on local torch 2.11.0 vs
  cluster's 2.5.1 — re-run the seconds-long correctness assert on-cluster
  before real sequence-arm GPU runs. No GPU/cluster resources used.
- **exp-001 (gate bundle): infra PASS; headline #1 FAIL on the C2 conjunct
  ONLY — banked as the run's core mechanistic finding.** #6 integration
  smoke PASS (loss decreases; filter-off == plain AdamW to fp64 precision;
  streaming variant runs; C1 spiked-covariance tests correct in hard mode).
  #9 data/protocol PASS: Numerai v5 downloaded, F1 split = tuning 0579-0966
  (388 eras) / embargo 0967-0970 / verdict 0971-1225 (255 eras), F5 embargo
  4 eras, F9 recalibrated sanity band [0.0087, 0.0522] and leakage gate
  0.0696. #7 throughput PASS (31 ms/step filter-on at B=1024; job 6430,
  2.5 min). #1: selectivity sustained (0<k<B on 100% of logged steps,
  kept-energy 0.29-0.89) and C3 clean (median cosF 0.25-0.83) — the
  pre-registered HARD fail-fast trigger did NOT fire. But real-vs-permuted-
  target diagnostics are indistinguishable, with a PROOF: for scalar-output
  MSE, per-sample gradients are +/-(residual sign) x a target-independent
  Jacobian direction, so S(y')=D S(y) D with D orthogonal — the whole
  engagement eigenspectrum is exactly target-independent at fixed parameters
  (verified fp64 to 8.9e-16). 'Consensus' on this task measures
  Jacobian/factor structure, not target signal. Binding registrations made
  before any OOS peek: F2 mode = hard (mp_factor 2.0), F6 = within-era
  composition. C5 probe inconclusive (carried as limitation). Surprise: the
  filter often AMPLIFIES (ratio > 1 on 25-70% of steps), it is not shrinkage.
- **ORCHESTRATOR DECISION (exp-001 middle-case ruling, for the Step 10
  audit)**: headline #1 logged FAIL (C2 conjunct failed by proof, not
  noise), but the binding hard fail-fast trigger did not fire and the
  motivating question remained open and answerable. Proceeded to exp-004
  under an AMENDED INTERPRETATION FRAME (in exp-004/plan.md): the
  target-independence proof is a co-primary deliverable; any verdict is
  attributed to target-blind spectral-subspace regularization, never
  'consensus on signal'; the C4 random-subspace control is load-bearing.
  All pre-registered verdict machinery unchanged (F1 first-unblinding, F2
  hard mode, F3=0.00398, F5 L=4, F8, F10, F12).
- **exp-003 (baseline sanity + power + budget + F4 gate): PASS on all
  sub-components** (job 6432, 1.1 min). #8: 12 AdamW trials on tuning-block
  eras only; best t07 (lr 1e-3, wd 1e-3, dropout 0.2) mean per-era
  numerai_corr +0.0196 INSIDE the F9 band; zero-predictor ~+0.0003; leakage
  gate NOT fired. Two extra seeds: +0.0129, +0.0153. #3 power PASS: F5
  moving-block bootstrap (L=4), paired 3-seed design at 255 verdict eras
  gives median CI half-width 0.00219; detection ~0.90-0.92 at +/-threshold.
  F3 FROZEN at 0.00398 (0.25 x 3-seed mean +0.01594, winner's-curse-
  corrected). F4 gate: P(any verdict category reachable) = 0.910 >= 0.6 ->
  GO. #4 budget PASS, no F7 cuts: exp-004 packs into one ~12.7-min job;
  exp-005 fits at ~13.7 min/task. F12 signatures RECORDED (binding): wd and
  dropout best at grid boundary with improving trends; spectral arm affords
  ~1 tuning trial vs AdamW's 12.
- **exp-004 (THE VERDICT EXPERIMENT): PASS — verdict returned** (job 6439,
  5.4 min on one L40; local block-bootstrap analysis in
  exp-004/out/verdict_analysis.json; full write-up in exp-004/results.md).
  First and only unblinding of the 255 verdict eras; all arms/configs frozen
  beforehand. **Pre-registered category: HURTS. Mandatory F12 downgrade
  applied: "no evidence of benefit under the affordable tuning budget."**
  Headline (filter_on − filter_off): −0.00527 mean per-era numerai_corr,
  95% CI [−0.00886, −0.00181] (spearman −0.00607 [−0.00927, −0.00292]);
  all three per-seed diffs negative; corr-Sharpe diff −0.200 [−0.416,
  −0.004]. Arm levels: filter_off +0.00641, filter_on +0.00113, C4 random
  subspace +0.00229, GAF −0.00042, zero-pred +0.00063. **Mechanism
  attribution (load-bearing): filter_on vs C4 = −0.00116 CI [−0.00366,
  +0.00134] — MP-eigenselection statistically indistinguishable from a
  norm/k-matched random subspace**; all three filtered arms hurt similarly
  (agreement-style update modification as a class). F8 tail breakdown:
  monotone helps-on-worst-eras / hurts-on-best-eras gradient (Q1 +0.019 →
  Q4 −0.0297) but flagged as regression-to-the-mean-contaminated
  (outcome-conditioned bucketing); the unconditioned target-dispersion
  split shows uniform hurt — no Feldman-tail pattern. Engagement confirmed
  (selective on 100% of steps; k_med 1-4; update norm ~10x mean gradient —
  amplification, not shrinkage). F9: verdict-block baseline +0.00641 is
  BELOW the tuning-band [0.0087, 0.0522] — regime drift, reported;
  leakage gate not fired; F3-vs-realized-baseline calibration gap (~62%
  relative bar vs intended 25%) carried as a limitation.
- **exp-005 (sequence-arm architecture consistency): PASS** (job 6614,
  6.6 min on one L40; both authorization conditions held). GRU (hand-rolled
  functional cell; within-era Numerai framing, T=15 x D=47 reshape of the
  705 features — F13 'architecture consistency' claim supported, OHLCV
  unused). On-cluster torch 2.5.1 correctness assert passed (4.77e-07).
  **The MLP verdict REPLICATES on the GRU: same category (HURTS → F12
  downgrade), headline −0.00484 CI [−0.00758, −0.00211]; spectral-vs-random
  null replicates (−0.00011 CI [−0.00233, +0.00215]); corr-Sharpe −0.242
  [−0.414, −0.073]; no dispersion-tail pattern.** Notable: the filter
  engages in a different regime on the GRU (k~60, mild attenuation) vs the
  MLP (k~1, ~10x amplification) yet produces near-identical harm and the
  same null — strengthening the 'generic subspace projection/rescaling,
  not spectral selection' account. Limitation: GRU used the t07 fallback
  config — the tuning-shard 'wiped' determination was made from the login
  node but /ephemeral is node-local, so the premise was false on the
  execution node; config was frozen pre-unblinding and identical across
  arms (protocol-valid), ~10 GPU-min re-tuning ask carried to Future Work.

**Run-level answer to the motivating question** (for Steps 10-11): the
Spectral Optimizer does NOT reduce the effect of noise on this large noisy
financial dataset. Under the pre-registered protocol it significantly
degraded out-of-sample performance on BOTH architectures (MLP and GRU),
with the F12-mandated conservative wording "no evidence of benefit under
the affordable tuning budget". The run's co-primary mechanistic
contributions: (1) a proof that for scalar-output MSE the filter's
eigenselection is exactly target-independent, and (2) matched controls
showing MP-eigenselection is empirically indistinguishable from random
subspace projection at the same kept-k/norm — together explaining WHY the
label-noise 'coherence amplifier' story does not transfer to low-SNR
financial regression. Carried limitations for the audit/report: C5 probe
inconclusive; F3 calibration gap under regime drift; GRU config fallback;
F8 baseline-quartile split is descriptive only; single spectral operating
point (affordability asymmetry, F12).

Step 9 complete. Next: Step 10 (results audit).

## Step 10: Results Audit — COMPLETE (round 1, HONEST-NEGATIVE exit)

Fresh results-auditor dispatched (round 1 of max 3). Full report:
`audit/results-audit.md`; audit artifacts in `audit/rerun-exp-004/`;
round-1 claim anchors frozen at `audit/claim-anchor-exp-00*.md`.

**Disposition: HONEST-NEGATIVE. audit_exit_reason: true-null. No
remediation round — exit to Step 11.** No finding could flip the verdict;
looping further would be re-confirming a null.

What the audit did:
- **Re-executed the load-bearing exp-004** on the cluster (producer's frozen
  code, fresh seeds {10,11,12}, plus an auditor-constructed verdict shard
  independently subsampled from the raw parquet; jobs 6616/6618/6620, ~9 min
  L40 total). The HURTS headline reproduces on both shards (-0.00546 /
  -0.00562 vs original -0.00527; 6-seed pool -0.00537 CI [-0.00853,
  -0.00222]); the C4 spectral-vs-random null replicates; the producer's
  shard row-matches the raw parquet (no fabrication).
- **Independently re-verified the target-independence proof** (different
  construction, 2.1e-14) and added two scoping controls (unnormalized
  spectrum and 2-output MSE are target-dependent) proving the theorem
  non-vacuous and correctly scoped to the deployed code path.
- Re-derived every results.md number from run.log/out/ artifacts with
  independent bootstrap code; verified F3=0.00398 arithmetic,
  freeze-before-unblinding via git commit ordering, F1 split assertions,
  no leakage via normalization constants, no grader-gaming signatures.

Findings (8): 5 SUPPORTED, 2 FIXABLE-DEFECT (write-up-level only), 1
TRUE-NULL. Notable:
- **NEW (found only by re-execution)**: the unmodified producer code crashes
  stochastically by seed (`linalg.eigh` non-convergence, fp32 cuSOLVER,
  under the rank-collapse regime the study itself documents). Original runs
  were 6-of-6-seed lucky-complete. Disclosure required in the paper;
  results unaffected (mathematically-equivalent CPU-fp64 fallback reproduced
  everything; patch preserved in audit/rerun-exp-004/src/).
- MLP corr-Sharpe CI upper bound is bootstrap-RNG fragile ([-0.416, -0.004]
  vs auditor's [-0.416, +0.003]) — report as marginal; GRU Sharpe
  (-0.242 CI [-0.414, -0.073]) is robust and carries the stability claim.
- Regime drift (TRUE-NULL): verdict-block baseline +0.0064 below the F9
  sanity band; scope the claim to this low-edge subsampled regime.
- exp-001 middle-case ruling examined for criteria drift and CLEARED
  (conservative, pre-unblinding, claim-narrowing); the paper must keep
  exp-001's headline as a logged FAIL-on-C2 mechanistic finding.

Step 11 (report) obligations from the audit:
- Include ALL 8 "unresolved findings for the write-up" in Limitations (and
  the email): regime drift/below-band baseline; F12 affordability asymmetry
  (keep the downgraded wording everywhere); untuned GRU config (~10-GPU-min
  ask); C5 probe inconclusive; eigh fragility; marginal MLP Sharpe; C4
  "indistinguishable" = equivalence at +/-0.004 resolution; F8 quartile
  split descriptive-only.
- Use the 15-row limitation-triage table (6 fix-now-free / 3 fix-now-cheap /
  6 future-work with concrete resource asks) for Limitations dispositions
  and the Future Work section.
- Robustness paragraph should cite the audit re-execution (6-seed pool,
  audit-shard replication) as the fix-now-free strengthening of the 3-seed
  headline.

Step 10 complete. Next: Step 11 (compile research report).

## Step 11: Research Report — COMPLETE

Report agent compiled the full LaTeX paper: `paper/paper.pdf` (16 pages, A4,
tectonic; all cross-references and all 28 citations resolve). Sources in
`paper/` (paper.tex, preamble.tex, references.bib, Makefile, sections/
abstract|introduction|related-work|methodology|experiments|discussion|
limitations|future-work|conclusion, figures/headline_forest.pdf generated
from the run's MBB estimates).

Title: "No Evidence of Benefit from Marchenko–Pastur Spectral Gradient
Filtering on Noisy Financial Time Series: A Pre-Registered Negative Result
with a Target-Independence Proof".

Audit obligations honored in the paper:
- F12 downgraded wording used everywhere ("no evidence of benefit under the
  affordable tuning budget"), raw hurts numbers always alongside — never a
  bare "hurts" headline.
- Results opens with the headline table (incl. audit re-runs and 6-seed
  pool, C4 nulls, corr-Sharpe with the MLP CI worded as marginal and the
  GRU carrying stability) plus a forest-plot summary figure.
- exp-001 kept as a logged FAIL-on-C2 mechanistic finding; the
  target-independence result stated as a theorem with the auditor's
  independent re-verification and scoping controls.
- All 8 unresolved audit findings in Limitations, each with an explicit
  triage disposition (addressed-by-wording / addressed-by-disclosure /
  deferred), including the linalg.eigh lucky-complete disclosure; the
  audit exit reason (true-null) stated.
- Future Work is a 7-item resource-scoped plan built from the 6 future-work
  triage rows plus design-triage W-items (headline: full-scale B~4096
  verdict run at ~10-20 L40-hours / ~$50-100 cloud; closure bundle ~25 min
  on one L40 covering GRU re-tune, FNC, C5 probe, extra C4 seed pairs).

`briefing.md` written for the interactive review command (topic, literature,
novelty, lambda table with outcomes, challenge highlights, results table
with the why-it-doesn't-work synthesis, abstract, surprises, follow-ups).

Run complete: honest-negative deliverable with two co-primary mechanistic
contributions (target-independence proof; spectral-vs-random equivalence).
