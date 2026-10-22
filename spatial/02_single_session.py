"""Prototype on one session: raw trajectory with spikes, rate maps, autocorrelograms, HD tuning."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spatial_lib import (autocorr2d, gridness, hd_curves, list_sessions, load_session, mean_vector, rate_maps,
                         spatial_info)

PATH = "sub-11265/sub-11265_ses-01020602_behavior+ecephys.nwb"

units, position, head_dir, moving, meta = load_session(dict(list_sessions())[PATH])
print(meta, f"{len(units)} units, moving {moving.tot_length() / meta['duration']:.0%} of the time")

# --- raw data: position, head direction, spike raster over the first 2 minutes ---
win = position.time_support.start[0] + np.array([0, 120])
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
p = position.get(*win)
axes[0].plot(p.t, p["x"], label="x")
axes[0].plot(p.t, p["y"], label="y")
axes[0].set_ylabel("position (cm)")
axes[0].legend(loc="upper right", ncol=2)
h = head_dir.get(*win)
axes[1].plot(h.t, np.degrees(h.values), ".", ms=1, color="k")
axes[1].set_ylabel("head dir. (deg)")
for k, u in enumerate(units.index):
    s = units[u].get(*win)
    axes[2].vlines(s.t, k + 0.6, k + 1.4, lw=0.6)
axes[2].set_yticks(range(1, len(units) + 1), units.metadata["unit_name"].values, fontsize=7)
axes[2].set_xlabel("time (s)")
axes[2].set_title("spikes", fontsize=9)
fig.suptitle(f"DANDI:000582 rat {meta['subject_id']} session {meta['session_id']}: raw behaviour and spikes")
fig.tight_layout()
fig.savefig("figures/01_raw_data.png", dpi=120)

# --- per-unit analysis ---
maps, occ = rate_maps(units, position, moving, meta["arena"])
curves, centers = hd_curves(units, head_dir, moving)
mvl, pref = mean_vector(curves, centers)
si = spatial_info(maps, occ)
acs = [autocorr2d(m) for m in maps]
gs = [gridness(a) for a in acs]

n = len(units)
fig = plt.figure(figsize=(2.2 * n, 9))
pmov = position.restrict(moving)
for k, u in enumerate(units.index):
    sp = units[u].restrict(moving)
    sxy = sp.value_from(pmov)
    ax = fig.add_subplot(4, n, k + 1)
    ax.plot(pmov["x"], pmov["y"], color="0.8", lw=0.3)
    ax.plot(sxy["x"], sxy["y"], ".", color="tab:red", ms=1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(units.metadata["unit_name"].iloc[k], fontsize=9)
    ax = fig.add_subplot(4, n, n + k + 1)
    ax.imshow(maps[k].T, origin="lower", cmap="jet", extent=[*meta["arena"][0], *meta["arena"][1]])
    ax.axis("off")
    ax.set_title(f"peak {np.nanmax(maps[k]):.1f} Hz\nSI {si[k]:.2f} b/s", fontsize=8)
    ax = fig.add_subplot(4, n, 2 * n + k + 1)
    ax.imshow(acs[k].T, origin="lower", cmap="jet", vmin=-0.5, vmax=1)
    ax.axis("off")
    ax.set_title(f"gridness {gs[k]:.2f}", fontsize=8)
    ax = fig.add_subplot(4, n, 3 * n + k + 1, projection="polar")
    ax.plot(np.r_[centers, centers[0]], np.r_[curves[k], curves[k][0]], color="tab:blue")
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["0", "90", "180", "270"], fontsize=6)
    ax.set_yticks([])
    ax.set_title(f"MVL {mvl[k]:.2f}", fontsize=8, pad=12)
fig.text(0.005, 0.87, "trajectory\n+ spikes", fontsize=9)
fig.text(0.005, 0.64, "rate map", fontsize=9)
fig.text(0.005, 0.40, "autocorr.", fontsize=9)
fig.text(0.005, 0.16, "head dir.", fontsize=9)
fig.suptitle(f"Rat {meta['subject_id']} session {meta['session_id']} (MEC layer III): every recorded unit", y=0.995)
fig.tight_layout(rect=(0.04, 0, 1, 0.98))
fig.savefig("figures/02_single_session_units.png", dpi=120)
