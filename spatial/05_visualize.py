"""Classify units (grid / head-direction / conjunctive) against shuffle thresholds and make summary figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/units.csv", dtype={"session": str, "subject": str})
z = np.load("results/maps.npz", allow_pickle=True)
maps, acs, curves, shuf = z["maps"], z["acs"], z["curves"], z["shuffles"]
centers = np.linspace(0, 2 * np.pi, curves.shape[1], endpoint=False) + np.pi / curves.shape[1]

# thresholds: 99th percentile of the shuffle distribution pooled over all units (Sargolini et al. 2006 used
# the same pooled-shuffle approach)
thr = {name: np.nanpercentile(shuf[:, i, :], 99) for i, name in enumerate(["spatial_info", "gridness", "mvl"])}
print("thresholds:", {k: round(v, 3) for k, v in thr.items()})

df["grid"] = df.gridness > thr["gridness"]
df["hd"] = df.mvl > thr["mvl"]
df["spatial"] = df.spatial_info > thr["spatial_info"]
df["cls"] = np.select(
    [df.grid & df.hd, df.grid, df.hd, df.spatial],
    ["conjunctive grid x HD", "pure grid", "pure head direction", "other spatial"],
    "not classified",
)
df.loc[~df.has_hd & ~df.grid, "cls"] = np.where(df.spatial[~df.has_hd & ~df.grid], "other spatial", "not classified")
df.loc[~df.has_hd & df.grid, "cls"] = "grid (no HD data)"
df.to_csv("results/units_classified.csv", index=False)

COLORS = {"pure grid": "#1b9e77", "conjunctive grid x HD": "#7570b3", "pure head direction": "#d95f02",
          "grid (no HD data)": "#66a61e", "other spatial": "#a6a6a6", "not classified": "#e0e0e0"}

summary = pd.crosstab(df.layer.replace("", np.nan).fillna("unknown"), df.cls)
summary["total"] = summary.sum(1)
print(summary)
summary.to_csv("results/class_by_layer.csv")
print(df.cls.value_counts())
n_sess = df.path.nunique()
n_rats = df.subject.nunique()

# ---------- Figure 3: population metrics vs shuffles ----------
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (i, key, label) in zip(axes[:2], [(1, "gridness", "gridness score"), (2, "mvl", "head-direction mean vector length")]):
    data = df[key].dropna()
    bins = np.linspace(np.nanmin(shuf[:, i]), max(np.nanmax(shuf[:, i]), data.max()), 50)
    ax.hist(shuf[:, i].ravel()[np.isfinite(shuf[:, i].ravel())], bins, density=True, color="0.7", label="shuffled")
    ax.hist(data, bins, density=True, histtype="step", lw=2, color="k", label="recorded units")
    ax.axvline(thr[key], color="r", ls="--", label="99th pct. of shuffle")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
ax = axes[2]
for c, col in COLORS.items():
    s = df[df.cls == c]
    ax.scatter(s.gridness, s.mvl, s=10, color=col, label=f"{c} ({len(s)})", edgecolor="none")
ax.axvline(thr["gridness"], color="r", ls="--", lw=1)
ax.axhline(thr["mvl"], color="r", ls="--", lw=1)
ax.set_xlabel("gridness score")
ax.set_ylabel("mean vector length")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.delaxes(axes[3])
ax = fig.add_subplot(1, 4, 4, projection="polar")
hd_cells = df[df.hd]
ax.hist(hd_cells.pref_dir, bins=24, color=COLORS["pure head direction"])
ax.set_title(f"preferred directions of {len(hd_cells)} HD-tuned cells", fontsize=9, pad=14)
ax.set_yticks([])
fig.suptitle(f"All {len(df)} units, {n_sess} sessions, {n_rats} rats (DANDI:000582)")
fig.tight_layout()
fig.savefig("figures/03_population_metrics.png", dpi=120)

# ---------- Figure 4: cell classes by layer ----------
order = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
cls_order = ["pure grid", "grid (no HD data)", "conjunctive grid x HD", "pure head direction", "other spatial",
             "not classified"]
frac = summary.reindex(order)[[c for c in cls_order if c in summary]].div(summary.reindex(order)["total"], axis=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
left = np.zeros(len(order))
for c in frac.columns:
    ax.barh(order, frac[c], left=left, color=COLORS[c], label=c)
    left += frac[c].values
for k, l in enumerate(order):
    ax.text(1.01, k, f"n={summary.loc[l, 'total']}", va="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("fraction of units")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_title("Functional cell classes by MEC layer")
fig.tight_layout()
fig.savefig("figures/04_classes_by_layer.png", dpi=120)


# ---------- Figure 5: example cells ----------
def examples(mask, key, n=4):
    return df[mask].sort_values(key, ascending=False).index[:n]


rows = [("pure grid", examples(df.cls == "pure grid", "gridness")),
        ("conjunctive grid x HD", examples(df.cls == "conjunctive grid x HD", "gridness")),
        ("pure head direction", examples(df.cls == "pure head direction", "mvl"))]
fig = plt.figure(figsize=(15, 7.5))
outer = fig.add_gridspec(len(rows), 4, hspace=0.55, wspace=0.15)
for r, (name, idx) in enumerate(rows):
    for c, i in enumerate(idx):
        inner = outer[r, c].subgridspec(1, 3, wspace=0.08)
        ax = fig.add_subplot(inner[0])
        ax.imshow(maps[i].T, origin="lower", cmap="jet", extent=[0, df.box_cm[i], 0, df.box_cm[i]])
        ax.axis("off")
        ax.set_title(f"{np.nanmax(maps[i]):.0f} Hz", fontsize=8)
        ax = fig.add_subplot(inner[1])
        ax.imshow(acs[i].T, origin="lower", cmap="jet", vmin=-0.5, vmax=1)
        ax.axis("off")
        ax.set_title(f"g={df.gridness[i]:.2f}", fontsize=8)
        ax = fig.add_subplot(inner[2], projection="polar")
        ax.plot(np.r_[centers, centers[0]], np.r_[curves[i], curves[i][0]], color=COLORS[name])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"MVL={df.mvl[i]:.2f}", fontsize=8, pad=4)
        if c == 0:
            fig.text(0.07, outer[r, 0].get_position(fig).y1 + 0.035, name, fontsize=12, weight="bold",
                     color=COLORS[name])
        fig.text(outer[r, c].get_position(fig).x0, outer[r, c].get_position(fig).y0 - 0.02,
                 f"rat {df.subject[i]} ses {df.session[i]} {df.unit[i]} ({df.layer[i]})", fontsize=7)
fig.suptitle("Example cells: rate map | spatial autocorrelogram | head-direction tuning", y=0.97)
fig.savefig("figures/05_example_cells.png", dpi=120, bbox_inches="tight")

# ---------- Figure 6: population decoding ----------
dec = pd.read_csv("results/decoding.csv", dtype={"session": str, "subject": str})
ex = np.load("results/decoding_example.npz")
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
def gapped(t, y):
    """Insert NaNs where held-out periods are not contiguous so lines are not drawn across gaps."""
    br = np.where(np.diff(t) > 1.0)[0] + 1
    return np.insert(t.astype(float), br, np.nan), np.insert(y.astype(float), br, np.nan)


ax = fig.add_subplot(gs[0, :2])
t, d, tr = ex["hd_0_t"], ex["hd_0_dec"], ex["hd_0_true"]
seg = t < t[0] + 60  # first held-out minute
ax.plot(t[seg], np.degrees(tr[seg]), ".", ms=3, color="k", label="actual heading")
ax.plot(t[seg], np.degrees(d[seg]), ".", ms=3, color=COLORS["pure head direction"], label="decoded from spikes")
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_ylim(-10, 420)
ax.set_yticks([0, 90, 180, 270, 360])
ax.legend(fontsize=8, loc="upper right", ncol=2)
ex_row = dec[dec.path == str(ex["path"])].iloc[0]
ax.set_title(f"Held-out heading decoded from {ex_row.n_units} cells (rat {ex_row.subject}, ses {ex_row.session}); "
             f"median error {ex_row.hd_err:.0f} deg vs {ex_row.hd_chance:.0f} deg chance", fontsize=10)
ax = fig.add_subplot(gs[1, :2])
t, d, tr = ex["pos_0_t"], ex["pos_0_dec"], ex["pos_0_true"]
seg = t < t[0] + 60
ax.plot(*gapped(t[seg], tr[seg, 0]), color="k", lw=1, label="actual x")
ax.plot(t[seg], d[seg, 0], ".", ms=3, color=COLORS["pure grid"], label="decoded x")
ax.set_ylabel("x position (cm)")
ax.set_ylim(-55, 70)
ax.set_xlabel("time (s)")
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.set_title(f"Held-out x position from the same cells; median 2D error {ex_row.pos_err:.0f} cm vs "
             f"{ex_row.pos_chance:.0f} cm chance", fontsize=10)
for r, (key, unit, label) in enumerate([("hd", "deg", "head direction"), ("pos", "cm", "position")]):
    ax = fig.add_subplot(gs[r, 2])
    s = dec.dropna(subset=[f"{key}_err"])
    better = s[f"{key}_err"] < s[f"{key}_chance_p05"]
    ax.scatter(s[f"{key}_chance"], s[f"{key}_err"], c=np.where(better, "tab:blue", "0.6"), s=12 + 3 * s.n_units)
    lim = [0, max(s[f"{key}_chance"].max(), s[f"{key}_err"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"chance error ({unit})")
    ax.set_ylabel(f"decoding error ({unit})")
    ax.set_title(f"{label}: {better.sum()}/{len(s)} sessions beat chance", fontsize=10)
fig.suptitle("Bayesian population decoding (fit on even minutes, tested on odd minutes and vice versa)")
fig.savefig("figures/06_decoding.png", dpi=120, bbox_inches="tight")

with open("results/summary.txt", "w") as f:
    f.write(f"units: {len(df)}, sessions: {n_sess}, rats: {n_rats}\n")
    f.write(f"thresholds (99th pct pooled shuffle): {thr}\n\n{summary.to_string()}\n\n")
    for key in ["hd", "pos"]:
        s = dec.dropna(subset=[f"{key}_err"])
        f.write(f"{key} decoding: {len(s)} sessions (>=5 units), median error {s[f'{key}_err'].median():.1f} vs chance "
                f"{s[f'{key}_chance'].median():.1f}; beat 5th pct of chance in "
                f"{(s[f'{key}_err'] < s[f'{key}_chance_p05']).sum()}\n")
print(open("results/summary.txt").read())
