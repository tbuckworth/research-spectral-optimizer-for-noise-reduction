#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
runs = json.load(open(ROOT/"out"/"paired-loss-curves.json"))["runs"]
colors = {"AdamW":"#2878b5", "StableSpectral":"#e36a22"}
labels = {"AdamW":"AdamW", "StableSpectral":"Corrected spectral"}
plt.style.use("seaborn-v0_8-whitegrid")

# All five seeds share the first 2,301 steps. Plot the mean EMA and seed range.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
common = 2301
for arm in colors:
    curves = np.array([r["ema_0.98"][:common] for r in runs if r["arm"]==arm])
    x = np.arange(1, common+1)
    axes[0].plot(x, curves.mean(0), color=colors[arm], lw=2.5, label=labels[arm])
    axes[0].fill_between(x, curves.min(0), curves.max(0), color=colors[arm], alpha=.12)
axes[0].axvline(100, color="black", ls="--", lw=1.5, label="filter activates")
axes[0].set_xlim(1, common); axes[0].set_ylim(.0475, .058)
axes[0].set(xlabel="Training step", ylabel="EMA minibatch MSE (decay 0.98)",
            title="Optimization trajectories overlap")
axes[0].legend()

# Fixed monitor: average the exact common checkpoints over seeds, omitting the
# first 50 steps where random initial outputs dominate the useful scale.
for arm in colors:
    selected = [r for r in runs if r["arm"]==arm]
    by_step = {}
    for r in selected:
        for z in r["fixed_train_mse"]:
            if 50 <= z["step"] <= common:
                by_step.setdefault(z["step"], []).append(z["mse"])
    steps = sorted(k for k,v in by_step.items() if len(v)==5)
    mean = np.array([np.mean(by_step[s]) for s in steps])
    sem = np.array([np.std(by_step[s],ddof=1)/np.sqrt(5) for s in steps])
    axes[1].plot(steps, mean, marker="o", ms=4, color=colors[arm], lw=2, label=labels[arm])
    axes[1].fill_between(steps, mean-sem, mean+sem, color=colors[arm], alpha=.15)
axes[1].set(xlabel="Training step", ylabel="MSE on fixed 65,536-row training monitor",
            title="Fixed-sample training loss is also indistinguishable")
axes[1].set_ylim(.0490, .0502); axes[1].legend()
fig.tight_layout(); fig.savefig(ROOT/"figures"/"paired_loss_curves.png", dpi=180); plt.close(fig)

# Paired robust late-training summaries.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, metric, title in [
    (axes[0], "last_500_mean", "Last-500 minibatch mean"),
    (axes[1], "fixed_late", "Mean of last five fixed-monitor checks")]:
    for seed in range(5):
        vals=[]
        for arm in colors:
            r=next(q for q in runs if q["arm"]==arm and q["seed"]==seed)
            value = r[metric] if metric!="fixed_late" else np.mean([z["mse"] for z in r["fixed_train_mse"][-5:]])
            vals.append(value)
        ax.plot([0,1], vals, color="#999", lw=1.8)
        ax.scatter([0,1], vals, color=[colors["AdamW"],colors["StableSpectral"]], s=65, zorder=3)
    ax.set_xticks([0,1],["AdamW","Corrected spectral"]); ax.set_ylabel("Training MSE"); ax.set_title(title)
fig.suptitle("No systematic terminal optimization gap", y=1.02, fontsize=17)
fig.tight_layout(); fig.savefig(ROOT/"figures"/"paired_terminal_losses.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# Direct differences reveal deviations hidden by the overlapping absolute curves.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ema_diffs=[]
for seed in range(5):
    a=next(r for r in runs if r["arm"]=="AdamW" and r["seed"]==seed)["ema_0.98"][:common]
    b=next(r for r in runs if r["arm"]=="StableSpectral" and r["seed"]==seed)["ema_0.98"][:common]
    d=np.asarray(b)-np.asarray(a); ema_diffs.append(d)
    axes[0].plot(np.arange(101,common+1),d[100:],color="#aaa",lw=.8,alpha=.6)
ema_diffs=np.asarray(ema_diffs)
axes[0].plot(np.arange(101,common+1),ema_diffs[:,100:].mean(0),color="#b42318",lw=2.5,label="five-seed mean")
axes[0].axhline(0,color="black",lw=1.3); axes[0].set_ylim(-.001,.001)
axes[0].set(xlabel="Training step",ylabel="Spectral − AdamW EMA minibatch MSE",
            title="Direct paired optimization-loss difference")
axes[0].legend()

fixed_maps=[]
for seed in range(5):
    a=next(r for r in runs if r["arm"]=="AdamW" and r["seed"]==seed)["fixed_train_mse"]
    b=next(r for r in runs if r["arm"]=="StableSpectral" and r["seed"]==seed)["fixed_train_mse"]
    aa={z["step"]:z["mse"] for z in a}; bb={z["step"]:z["mse"] for z in b}
    fixed_maps.append({s:bb[s]-aa[s] for s in set(aa)&set(bb) if 100<=s<=common})
steps=sorted(set.intersection(*(set(m) for m in fixed_maps)))
fixed_by_seed=[[m[s] for s in steps] for m in fixed_maps]
fixed_by_seed=np.asarray(fixed_by_seed); mean=fixed_by_seed.mean(0); sem=fixed_by_seed.std(0,ddof=1)/np.sqrt(5)
axes[1].plot(steps,mean,marker="o",color="#7b3294",lw=2.2)
axes[1].fill_between(steps,mean-sem,mean+sem,color="#7b3294",alpha=.18)
axes[1].axhline(0,color="black",lw=1.3); axes[1].set_ylim(-.001,.001)
axes[1].set(xlabel="Training step",ylabel="Spectral − AdamW fixed-monitor MSE",
            title="Direct fixed-training-monitor difference")
fig.tight_layout(); fig.savefig(ROOT/"figures"/"paired_loss_differences.png",dpi=180); plt.close(fig)

summary={}
for metric in ("last_100_mean","last_500_mean"):
    diffs=[]
    for seed in range(5):
        a=next(r for r in runs if r["arm"]=="AdamW" and r["seed"]==seed)[metric]
        b=next(r for r in runs if r["arm"]=="StableSpectral" and r["seed"]==seed)[metric]
        diffs.append(b-a)
    summary[metric]={"mean_adamw":float(np.mean([r[metric] for r in runs if r['arm']=='AdamW'])),
                     "mean_spectral":float(np.mean([r[metric] for r in runs if r['arm']=='StableSpectral'])),
                     "mean_paired_difference":float(np.mean(diffs)),"per_seed_difference":diffs}
(ROOT/"out"/"loss-curve-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
