"""Quality checks and a raw-data view of one session.

- photometry sampling regularity, NaNs, session order per mouse
- trial structure: every CS+ followed by sucrose 1.25 s later
- figure: raw dF/F with cue, reward and lick marks for the first trials of a mouse's
  first session (figures/01_raw_session.png)
- figure: QC summary across sessions (figures/02_qc.png)

Output: data/sessions.csv (one row per included session, with cumulative trial index).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

HERE = Path(__file__).parent
DATA = HERE / "data"
FIG = HERE / "figures"
GROUP_ORDER = ["60sD", "600sD", "3600sD"]
COLORS = {"60sD": "#2a6fdb", "600sD": "#d9822b", "3600sD": "#3a9b5c"}


def load_session(row):
    """Return pynapple objects for one session."""
    s = np.load(DATA / row["file"])
    dff = nap.Tsd(t=s["dff_t"], d=s["dff"].astype(float))
    return dict(
        dff=dff,
        cs=nap.Ts(s["cs"]),
        reward=nap.Ts(s["reward"]),
        lick=nap.Ts(s["lick"]),
    )


def main():
    FIG.mkdir(exist_ok=True)
    sessions = pd.DataFrame(json.loads((DATA / "index.json").read_text()))

    qc = []
    for _, row in sessions.iterrows():
        s = np.load(DATA / row["file"])
        dt = np.diff(s["dff_t"])
        cs, rw = s["cs"], s["reward"]
        delay = rw - cs if len(cs) == len(rw) else np.array([np.nan])
        qc.append(
            dict(
                fs_hz=1 / np.median(dt),
                max_gap_s=dt.max(),
                n_nan=int(np.isnan(s["dff"]).sum()),
                dff_sd=float(np.nanstd(s["dff"])),
                cs_reward_delay_min=float(np.nanmin(delay)),
                cs_reward_delay_max=float(np.nanmax(delay)),
                cs_after_end=int((cs > s["dff_t"][-1] - 5).sum()),
            )
        )
    sessions = pd.concat([sessions, pd.DataFrame(qc)], axis=1)
    pd.set_option("display.width", 200)
    print(sessions.groupby("group")[["n_cs", "fs_hz", "max_gap_s", "n_nan", "cs_reward_delay_min", "cs_reward_delay_max"]].agg(["min", "max"]))
    print(sessions["session_description"].value_counts())
    print(sessions["fiber_location"].value_counts())

    # Session order within each mouse; "day02a"/"day02b" sort correctly as strings.
    sessions = sessions.sort_values(["group", "subject", "session"]).reset_index(drop=True)
    sessions["session_idx"] = sessions.groupby("subject").cumcount()
    sessions["first_trial"] = sessions.groupby("subject")["n_cs"].cumsum() - sessions["n_cs"]
    sessions.to_csv(DATA / "sessions.csv", index=False)
    print(sessions.groupby(["group", "subject"])["n_cs"].agg(["count", "sum"]))

    # Raw view of the first trials of the first 60 s ITI mouse.
    row = sessions[(sessions.group == "60sD")].iloc[0]
    sess = load_session(row)
    ep = nap.IntervalSet(start=sess["cs"].t[0] - 20, end=sess["cs"].t[4] + 20)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    d = sess["dff"].restrict(ep)
    ax.plot(d.t, d.d, color="0.25", lw=0.6, label="NAc dopamine (dF/F)")
    lk = sess["lick"].restrict(ep).t
    ytop = np.percentile(d.d, 99.9)
    ax.plot(lk, np.full_like(lk, ytop * 1.08), "|", color="#7b4fa0", ms=8, label="lick")
    for i, t in enumerate(sess["cs"].restrict(ep).t):
        ax.axvline(t, color=COLORS["60sD"], lw=1.2, label="CS+ onset (tone)" if i == 0 else None)
    for i, t in enumerate(sess["reward"].restrict(ep).t):
        ax.axvline(t, color="#c0392b", lw=1.2, ls="--", label="sucrose" if i == 0 else None)
    ax.set_xlabel("Time in session (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"{row['subject']}, {row['session']}: first 5 cue-sucrose pairings (naive mouse)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "01_raw_session.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for g in GROUP_ORDER:
        sg = sessions[sessions.group == g]
        axes[0].scatter(sg["session_idx"] + 1, sg["n_cs"], color=COLORS[g], s=12, label=g.replace("D", " ITI"))
        axes[1].scatter(sg["session_idx"] + 1, sg["dff_sd"], color=COLORS[g], s=12)
        axes[2].scatter(sg["session_idx"] + 1, sg["n_lick"] / sg["duration_s"] * 60, color=COLORS[g], s=12)
    axes[0].set_yscale("log")
    axes[0].set_yticks([2, 6, 11, 50], ["2", "6", "11", "50"])
    axes[0].minorticks_off()
    axes[0].set_ylabel("Cue-sucrose pairings")
    axes[1].set_ylabel("dF/F standard deviation")
    axes[2].set_ylabel("Licks per minute")
    for ax in axes:
        ax.set_xlabel("Session")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Quality checks across all sessions")
    fig.tight_layout()
    fig.savefig(FIG / "02_qc.png", dpi=150)


if __name__ == "__main__":
    main()
