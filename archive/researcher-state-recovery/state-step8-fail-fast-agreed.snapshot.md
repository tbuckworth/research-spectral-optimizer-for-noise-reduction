---
run_id: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
topic: "Spectral Optimizer (for noise reduction) on Financial Timeseries Data"
current_step: 8
status: fail_fast_agreed
mode: autonomous
agent_backend: claude
agent_model: "fable"
issue_number: 2073
compute_profile: "MATS Slurm cluster, driven REMOTELY over SSH from wherever this run is executing (laptop or desktop) - you are NOT on the cluster. Every cluster command is prefixed: ssh mats '<cmd>' for the dev/login node, ssh mats-controller '<cmd>' for the controller (the only place gpu-avail and gpu-cost exist). AUTHORIZED COMPUTE: the FREE 'compute' partition only - one shared always-on node with 8x NVIDIA L40 (48GB VRAM each). Your concurrent cap across all your jobs is 6 GPUs, 124 CPUs, and 384GB RAM; max wall time is 24h per job. Request 1 GPU (--gres=gpu:1) unless the experiment genuinely needs more, and up to 6 when it does (single node, so torchrun/FSDP works). Sizing on 48GB: fp16 inference up to ~20B params, LoRA/QLoRA fine-tuning up to ~7B (13B with care), full fine-tuning only up to ~2-3B. Add --qos=debug for validation runs under 2h to jump the queue. PAID elastic-* partitions (A100/H100) are NOT authorized and the account is not enabled for them: if an experiment needs more than 6 L40s or more VRAM per device, do not attempt it - record a FAIL-on-affordability with the exact resource ask and continue. WORKFLOW for each experiment: (1) write the code and an sbatch script locally under experiments/exp-NNN/; (2) stage it with rsync -avP experiments/exp-NNN/ mats:/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN/; (3) submit with ssh mats 'cd /mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-NNN && mkdir -p logs && sbatch run.sbatch' - create logs/ BEFORE sbatch, because Slurm opens the output file at job launch and an in-script mkdir is too late; (4) poll with ssh mats 'squeue -u t.buckworth' until the job leaves the queue (PD pending, R running, CG completing), sleeping between polls rather than busy-looping; (5) pull results back with rsync -avP mats:.../exp-NNN/ experiments/exp-NNN/ and copy the full slurm log into run.log. NEVER run training, fine-tuning, heavy inference, or long CPU loops in an ssh shell on the dev node - it is the shared login node and that is the cluster's worst etiquette violation. Never run agents or jobs on the controller. The local machine this run executes on is for orchestration, plotting, and light analysis of returned results only - do not silently fall back to a local GPU for experiments under this profile. Job script requirements: all #SBATCH lines must precede the first command (any command above them silently disables every directive below); include --partition=compute, --gres=gpu:1, a realistic --time, --job-name, --cpus-per-task=8, --mem=32G, --output=logs/slurm-%j.out. Storage on the cluster: code, checkpoints, and final results under /mnt/nw/home/t.buckworth (persistent NFS, NOT backed up - pull anything irreplaceable back to this machine); HuggingFace cache, dataset shards, and intermediate outputs under /ephemeral/t.buckworth (fast local scratch, wiped on reboot) via HF_HOME=/ephemeral/t.buckworth/hf. Inside every job script: source ~/venv/bin/activate (the cluster's shared venv) and run python with -u so logs are not buffered. That venv pins torch 2.5.1+cu121 to match the workers' CUDA 12.2 driver - do NOT upgrade torch or install one from default PyPI, which breaks CUDA on every job. Diagnostics: ssh mats 'scontrol show job <jobid>' explains a stuck job; sacct works only from the controller (it errors with Connection refused on the dev node); nvidia-smi on the dev node fails because there is no GPU there, which is expected; on compute, nvidia-smi inside a job lists all 8 physical GPUs but you only own $CUDA_VISIBLE_DEVICES; an empty .out file on a running job is usually stdout buffering. scancel anything left idle. Prefer lightweight experiments (small open-weight models), each targeting under 30 min of GPU time. Max 5 experiments."
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
lambda_table:
  - component: "Spectral engagement on financial gradients"
    p_success: 0.40
    t_hours: 0.75
    lambda: 1.22
    status: PENDING
  - component: "Sequence-arm vmap feasibility (recurrent model)"
    p_success: 0.35
    t_hours: 1.0
    lambda: 1.05
    status: PENDING
  - component: "Statistical power of paired per-era design"
    p_success: 0.60
    t_hours: 0.5
    lambda: 1.02
    status: PENDING
  - component: "Full design fits 5-experiment / 30-min budget"
    p_success: 0.80
    t_hours: 0.25
    lambda: 0.89
    status: PENDING
  - component: "OHLCV fallback dataset for sequence arm"
    p_success: 0.80
    t_hours: 0.5
    lambda: 0.45
    status: PENDING
  - component: "Spectral Optimizer regression integration"
    p_success: 0.75
    t_hours: 0.75
    lambda: 0.38
    status: PENDING
  - component: "vmap throughput on L40 within job cap"
    p_success: 0.85
    t_hours: 0.5
    lambda: 0.33
    status: PENDING
  - component: "Baseline sanity: tuned AdamW in sane corr band"
    p_success: 0.70
    t_hours: 1.5
    lambda: 0.24
    status: PENDING
  - component: "Numerai data + era-purged protocol"
    p_success: 0.85
    t_hours: 1.0
    lambda: 0.16
    status: PENDING
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

## Step 9: Experiment Execution (in progress)

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
- exp-001 (gate bundle): in progress. Local sub-components DONE and healthy:
  #6 integration smoke PASS (loss decreases; filter-off == plain AdamW to
  fp64 precision; streaming variant runs; C1 spiked-covariance tests correct
  in hard mode — iid noise keeps ~0.1% norm, planted spike kept at cos 0.96,
  confound case measured: hard mode keeps the correlated zero-signal confound,
  calibrating the 'engaged != meaningful' caveat). #9 data/protocol DONE:
  Numerai v5 downloaded (~6GB local), 643 usable validation eras, F1 split =
  tuning 0579-0966 (388 eras) / embargo 0967-0970 / verdict 0971-1225 (255
  eras), F5 embargo 4 eras (=ceil(20d/5d)), F9 recalibrated sanity band
  [0.0087, 0.0522] and leakage gate 0.0696 from current v5 example preds
  (out/protocol.json, out/f9_recalibration.json). Train shard (150 eras, 705
  medium features) staged on cluster. #7+#1 GPU debug job = Slurm job 6430,
  queued PD(Resources) since 08:18, being watched.
- exp-003: #8 GPU sweep SUBMITTED and COMPLETED as Slurm job 6432 (1.1 min on
  one L40). Raw outcome: 12 AdamW trials on the 388 tuning-block eras only;
  best trial t07 (lr 0.001, wd 0.001, dropout 0.2) mean per-era numerai_corr
  +0.0196 — INSIDE the F9 band [0.0087, 0.0522]; zero-predictor ~+0.0003;
  leakage gate NOT fired (no hard fail-fast). Two extra seeds of the best
  config: +0.0129, +0.0153 (winner's-curse shrinkage visible). Local analysis
  COMPLETE (results.md written): exp-003 = PASS on all sub-components.
  #3 power PASS: F5 moving-block bootstrap (L=4, from lag-1 acf +0.247 and
  the 4-era embargo), paired 3-seed design at 255 verdict eras gives median
  CI half-width 0.00219 <= F3; detection ~0.90-0.92 at +/-threshold. F3
  FROZEN at 0.00398 mean per-era numerai_corr (rule: 0.25 x the 3-seed mean
  +0.01594 of the best config — the conservative winner's-curse-corrected
  choice; sweep-selection alternative 0.00490 recorded but not used). F4
  gate: P(any verdict category reachable) = 0.910 >= 0.6 -> GO (sensitivity:
  at 2x measured seed noise it would read 0.096; that contingency lands in
  exp-004's pre-authorized no-verdict path). #4 budget PASS, no F7 cuts:
  exp-004 packs into ONE ~12.7-min GPU job (spectral tuning trial +
  filter-on x3 + filter-off x3 + C4 random-subspace x3 + GAF ablation x3,
  2000 steps @ B=1024); conditional exp-005 = 2-task array at ~13.7
  min/task. F12 signatures RECORDED (binding on exp-004 wording): wd and
  dropout best at grid boundary with improving trends; spectral arm affords
  ~1 tuning trial vs AdamW's 12 -> any 'hurts' verdict must be downgraded
  to 'no evidence of benefit under the affordable tuning budget'.
- exp-001 GPU debug job 6430 COMPLETED (2.5 min on one L40, 08:57). Raw
  outcome: #7 throughput healthy (31 ms/step filter-on at B=1024; projected
  2000-step run ~1 min — far under the 20-min cap, no F11 scope change
  needed). #1 engagement raw logs show selective k (4-57 of B) across all
  modes/batch sizes/compositions with varying C3 cosines; the pass/fail crux
  is C2 (permuted-target runs look qualitatively similar to real-target runs
  in the coarse log — quantitative distinguishability analysis required).
  Verdict analysis COMPLETE (results.md written): #6/#9/#7 all PASS;
  headline #1 = FAIL on the C2 conjunct ONLY. Selectivity sustained (0<k<B on
  100% of logged steps, kept-energy 0.29-0.89) and C3 clean (median cosF
  0.25-0.83; >0.95 on <=15% of steps) — the pre-registered HARD fail-fast
  trigger (all-modes degeneracy OR cosF>0.95 everywhere) did NOT fire. But
  real-vs-permuted-target diagnostics are indistinguishable, with a PROOF:
  for scalar-output MSE, per-sample gradients are +/-(residual sign) x a
  target-independent Jacobian direction, so the row-normalized similarity
  matrix satisfies S(y')=D S(y) D with D orthogonal — the whole engagement
  eigenspectrum is exactly target-independent at fixed parameters (verified
  fp64 to 8.9e-16; explains why MNIST cross-entropy in the prior project WAS
  target-sensitive). 'Consensus' on this task measures Jacobian/factor
  structure, not target signal — the core mechanistic finding of the run.
  Binding registrations made before any OOS peek: F2 mode = hard (mp_factor
  2.0; only mode passing C1 calibration), F6 = within-era composition. C5
  probe inconclusive (carried as limitation). Surprise: ratio =
  ||filtered||/||mean_grad|| exceeds 1 on 25-70% of steps — the filter often
  AMPLIFIES (discards anti-aligned components), it is not shrinkage.
- ORCHESTRATOR DECISION (exp-001 middle-case ruling, recorded for the Step
  10 audit): exp-001's headline #1 is logged FAIL because the pre-registered
  PASS criterion required C2 real-vs-permuted distinguishability, which
  failed (by proof, not by noise). However the BINDING hard fail-fast
  trigger in fail_fast_conditions (all-modes degeneracy OR C3>0.95
  everywhere) did NOT fire — the filter does sustained selective
  non-trivial work. The motivating question (does the optimizer improve OOS
  performance vs tuned AdamW?) therefore remains open and answerable: a
  target-blind spectral subspace projection can still help or hurt
  generalization, like any structured regularizer. DECISION: proceed to
  exp-004 under an AMENDED INTERPRETATION FRAME (recorded in
  exp-004/plan.md): the target-independence proof is a co-primary
  deliverable; any verdict is attributed to target-blind spectral-subspace
  regularization, never 'consensus on signal'; the C4 random-subspace
  control is load-bearing and protected. All pre-registered verdict
  machinery unchanged (F1 first-unblinding, F2 hard mode, F3=0.00398, F5
  L=4, F8, F10, F12). exp-001's mechanistic-pivot deliverable is BANKED
  regardless of exp-004's outcome.
- exp-004: plan.md written; experiment agent dispatched (~10:00) to build
  the verdict shard (eras 0971-1225, first unblinding), run ONE ~13-min GPU
  job (4 arms x 3 seeds), and compute the verdict locally.
- exp-005 decision deferred until exp-004 returns (exp-002 PASS and budget
  arithmetic both permit it; F13 naming rules apply).
- Session note: resumed Step 9 session #2 (2026-08-01 ~09:40). exp-001 and
  exp-003 results.md complete; exp-004 agent running.
