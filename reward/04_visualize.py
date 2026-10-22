"""Figures for cue-reward learning in DANDI:001632 dopamine cohorts.

figures/03_example_heatmaps.png  trial-by-trial dopamine and licking, one mouse per ITI group
figures/04_peth_by_session.png   mean dopamine and lick PETHs per session, per group
figures/05_learning_curves.png   anticipatory licking and dopamine cue response vs trials and vs time
figures/06_learned_trial_vs_iti.png  trials (and hours) to learn as a function of ITI
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"
FIG = HERE / "figures"
GROUP_ORDER = ["60sD", "600sD", "3600sD"]
LABEL = {"60sD": "60 s ITI", "600sD": "600 s ITI", "3600sD": "3600 s ITI"}
COLORS = {"60sD": "#2a6fdb", "600sD": "#d9822b", "3600sD": "#3a9b5c"}
DELAY = 1.25


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def mark_events(ax):
    ax.axvline(0, color="0.4", lw=0.8)
    ax.axvline(DELAY, color="#c0392b", lw=0.8, ls="--")


def example_heatmaps(trials, peth, lick):
    t, P = peth["t"], peth["peth"]
    edges = lick["lick_edges"]
    learning = pd.read_csv(DATA / "learning.csv").set_index("subject")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), gridspec_kw=dict(height_ratios=[1, 1]))
    for k, g in enumerate(GROUP_ORDER):
        # Mouse with the median dopamine learning trial in its group.
        lg = learning[learning.group == g].sort_values("learned_trial_da")
        subject = lg.index[len(lg) // 2]
        idx = trials.index[trials.subject == subject].values
        n = len(idx)
        im = axes[0, k].imshow(
            P[idx], aspect="auto", cmap="magma", vmin=-1, vmax=np.percentile(P[idx], 99),
            extent=[t[0], t[-1], n + 0.5, 0.5], interpolation="nearest",
        )
        fig.colorbar(im, ax=axes[0, k], label="dF/F (baseline-subtracted)", pad=0.02)
        L = lick["lick_counts"][idx] / 0.1
        im2 = axes[1, k].imshow(
            L, aspect="auto", cmap="Greys", vmin=0, vmax=10,
            extent=[edges[0], edges[-1], n + 0.5, 0.5], interpolation="nearest",
        )
        fig.colorbar(im2, ax=axes[1, k], label="Lick rate (Hz)", pad=0.02)
        missing = np.flatnonzero(trials.loc[idx, "anticipatory_lick_hz"].isna().values)
        if len(missing):
            axes[1, k].axhspan(missing[0] + 0.5, missing[-1] + 1.5, color="0.85", zorder=2)
            axes[1, k].text(2, (missing[0] + missing[-1]) / 2 + 1, "no licks logged (software crash)",
                            ha="center", va="center", fontsize=8, zorder=3)
        for row in range(2):
            ax = axes[row, k]
            mark_events(ax)
            ax.set_xlim(-2, 6)
            ax.axhline(learning.loc[subject, "learned_trial_da" if row == 0 else "learned_trial_lick"] - 0.5,
                       color="cyan", lw=1.4, ls=":")
            ax.set_ylabel("Trial (cumulative)")
        axes[0, k].set_title(f"{LABEL[g]}: {subject}\nNAc dopamine", fontsize=10)
        axes[1, k].set_title("Licking", fontsize=10)
        axes[1, k].set_xlabel("Time from CS+ onset (s)")
    fig.suptitle("Single mice across all of training (solid: cue, dashed: sucrose, dotted: learning trial)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "03_example_heatmaps.png", dpi=150)


def peth_by_session(trials, peth, lick):
    t, P = peth["t"], peth["peth"]
    edges = lick["lick_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for k, g in enumerate(GROUP_ORDER):
        tg = trials[trials.group == g]
        n_sess = int(tg.session_idx.max()) + 1
        cmap = plt.get_cmap("viridis")
        ymax = 0
        for s in range(n_sess):
            sel = tg[tg.session_idx == s]
            if sel.subject.nunique() < 3:
                continue
            # Average per mouse first, then across mice.
            per_mouse = np.stack([P[sel.index[sel.subject == m]].mean(0) for m in sel.subject.unique()])
            per_mouse_l = np.stack([lick["lick_counts"][sel.index[sel.subject == m]].mean(0) / 0.1 for m in sel.subject.unique()])
            c = cmap(s / max(n_sess - 1, 1))
            axes[0, k].plot(t, per_mouse.mean(0), color=c, lw=1.2, label=f"session {s + 1}")
            axes[1, k].plot(centers, per_mouse_l.mean(0), color=c, lw=1.2)
            ymax = max(ymax, per_mouse.mean(0).max())
        axes[0, k].set_title(f"{LABEL[g]} (n = {tg.subject.nunique()} mice, {int(tg.groupby('subject').size().median())} trials each)", fontsize=10)
        axes[0, k].set_ylabel("Dopamine dF/F")
        axes[1, k].set_ylabel("Lick rate (Hz)")
        axes[1, k].set_xlabel("Time from CS+ onset (s)")
        for row in range(2):
            mark_events(axes[row, k])
            tidy(axes[row, k])
        axes[0, k].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    axes[0, 0].set_xlim(-2, 6)
    fig.suptitle("Mean response on each training session (cue at 0 s, sucrose at 1.25 s)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "04_peth_by_session.png", dpi=150)


def binned_curve(tr, col, x, edges):
    """Mean of col per mouse inside x bins, then mean and sem across mice."""
    per_mouse = []
    for _, m in tr.groupby("subject"):
        b = np.digitize(m[x], edges) - 1
        per_mouse.append([m[col][b == i].mean() if np.any(b == i) else np.nan for i in range(len(edges) - 1)])
    per_mouse = np.array(per_mouse)
    n = np.sum(~np.isnan(per_mouse), axis=0)
    mean = np.nanmean(per_mouse, axis=0)
    sem = np.nanstd(per_mouse, axis=0) / np.sqrt(np.maximum(n, 1))
    mean[n < 3] = np.nan
    return mean, sem


def learning_curves(trials):
    trials = trials.copy()
    trials["hours"] = trials["trial"] * trials["mean_iti_s"] / 3600
    # Per-mouse z-normalized dopamine cue response, so mice with different signal sizes
    # contribute equally: divided by that mouse's mean reward response on session 1.
    norm = trials[trials.session_idx == 0].groupby("subject")["da_reward"].mean()
    trials["da_cue_norm"] = trials["da_cue"] / trials["subject"].map(norm)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for g in GROUP_ORDER:
        tg = trials[trials.group == g]
        n_trials = int(tg.trial.max()) + 1
        # Log-spaced trial bins so the 16-trial and 400-trial groups are both visible.
        edges_trial = np.unique(np.round(np.geomspace(1, n_trials + 1, 14))).astype(float) - 1
        edges_time = np.linspace(0, 16, 17)
        for row, col in enumerate(["anticipatory_lick_hz", "da_cue_norm"]):
            m, s = binned_curve(tg, col, "trial", edges_trial)
            xc = np.sqrt(np.maximum(edges_trial[:-1], 0) + 1) * np.sqrt(edges_trial[1:])
            axes[row, 0].errorbar(xc, m, yerr=s, color=COLORS[g], marker="o", ms=3, lw=1.2, capsize=0, label=LABEL[g])
            m, s = binned_curve(tg, col, "hours", edges_time)
            xc = (edges_time[:-1] + edges_time[1:]) / 2
            axes[row, 1].errorbar(xc, m, yerr=s, color=COLORS[g], marker="o", ms=3, lw=1.2, capsize=0, label=LABEL[g])
    for row in range(2):
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_xlabel("Cue-sucrose pairings experienced (log scale)")
        axes[row, 1].set_xlabel("Hours of conditioning (pairings x ITI)")
        for c in range(2):
            axes[row, c].axhline(0, color="0.6", lw=0.6)
            tidy(axes[row, c])
    axes[0, 0].set_ylabel("Anticipatory licking (Hz,\ndelay minus pre-cue)")
    axes[0, 1].set_ylabel("Anticipatory licking (Hz)")
    axes[1, 0].set_ylabel("Dopamine cue response\n(fraction of session-1 sucrose response)")
    axes[1, 1].set_ylabel("Dopamine cue response\n(normalized)")
    axes[0, 0].set_title("Learning vs number of pairings", fontsize=10)
    axes[0, 1].set_title("Learning vs time spent in conditioning", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean +/- SEM across mice", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "05_learning_curves.png", dpi=150)


def learned_vs_iti():
    learning = pd.read_csv(DATA / "learning.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rng = np.random.default_rng(0)
    jitter = np.exp(rng.uniform(-0.08, 0.08, len(learning)))
    criteria = [("", "elbow of cumulative curve"), ("_half", "running mean reaches half of asymptote")]
    for r, (suffix, crit) in enumerate(criteria):
        for c, (measure, name) in enumerate([("lick", "anticipatory licking"), ("da", "dopamine cue response")]):
            ax = axes[r, c]
            col = f"learned_trial_{measure}{suffix}"
            y = learning[col]
            ax.scatter(learning["mean_iti_s"] * jitter, y, c=[COLORS[g] for g in learning.group],
                       s=30, edgecolor="k", lw=0.4, zorder=3)
            med = learning.groupby("mean_iti_s")[col].median()
            ax.plot(med.index, med.values, "k_", ms=22, mew=2, zorder=4)
            fit = stats.linregress(np.log10(learning["mean_iti_s"]), np.log10(y))
            xx = np.array([40, 5000])
            ax.plot(xx, 10 ** (fit.intercept + fit.slope * np.log10(xx)), color="0.3", lw=1,
                    label=f"log-log fit: slope {fit.slope:.2f} "
                          f"[95% CI {fit.slope - 1.96 * fit.stderr:.2f}, {fit.slope + 1.96 * fit.stderr:.2f}]")
            ax.plot(xx, med.loc[60] * 60 / xx, color="0.6", lw=1, ls="--", label="slope -1 (trials x ITI constant)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([60, 600, 3600], ["60", "600", "3600"])
            ax.minorticks_off()
            ax.set_ylim(0.7, 400)
            ax.set_yticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"])
            ax.set_xlabel("Mean inter-trial interval (s)")
            ax.set_ylabel("Trials to learn")
            ax.set_title(f"{name.capitalize()}\ncriterion: {crit}", fontsize=10)
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            tidy(ax)
            print(f"{name} / {crit}: slope {fit.slope:.3f} +/- {fit.stderr:.3f}, r {fit.rvalue:.3f}, p {fit.pvalue:.2g}")
    fig.suptitle("Pairings needed to learn the cue fall as the time between rewards grows\n"
                 "(points: mice; black bars: group medians)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "06_learned_trial_vs_iti.png", dpi=150)


def main():
    FIG.mkdir(exist_ok=True)
    trials = pd.read_csv(DATA / "trials.csv")
    peth = np.load(DATA / "peth.npz")
    example_heatmaps(trials, peth, peth)
    peth_by_session(trials, peth, peth)
    learning_curves(trials)
    learned_vs_iti()


if __name__ == "__main__":
    main()
