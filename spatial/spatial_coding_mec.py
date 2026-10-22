# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How does an animal know where it is in space?
#
# To know where it is, an animal needs at least two internal signals: which way it is facing, and where it
# is relative to the environment. In the rodent medial entorhinal cortex (MEC) both signals appear in the
# firing of single neurons:
#
# - **Head-direction (HD) cells** fire when the head points in one direction, wherever the animal is.
# - **Grid cells** fire at several locations that form a triangular (hexagonal) lattice across the floor.
# - **Conjunctive grid x HD cells** fire at grid locations, but only when the animal faces one direction.
#
# Together these cells can support path integration: heading and running speed are combined to update a
# position estimate as the animal moves.
#
# This notebook tests these ideas on real recordings from **DANDI:000582**
# ([Sargolini et al., Science 2006](https://doi.org/10.1126/science.1125572)): tetrode recordings from MEC
# layers II, III, V and VI of rats foraging for food in open arenas, with two head-mounted LEDs giving
# position and heading. The steps are:
#
# 1. Look at raw behaviour and spikes from one session.
# 2. Compute rate maps, spatial autocorrelograms (gridness) and head-direction tuning for every unit in
#    all 118 sessions, with time-shift shuffles as a null distribution.
# 3. Classify cells and compare layers.
# 4. Decode the animal's heading and position from population activity on held-out data.
#
# Shared code lives in `spatial_lib.py`. The per-session workers come from `03_analyze_all_sessions.py`
# and `04_decode_position_heading.py`, and the summary figures from `05_visualize.py`.

# %%
import importlib.util
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from tqdm import tqdm

from spatial_lib import (autocorr2d, gridness, hd_curves, list_sessions, load_session, mean_vector, rate_maps,
                         spatial_info)

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_module(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3].lstrip("0123456789_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so worker processes can pickle its functions
    spec.loader.exec_module(mod)
    return mod


# %% [markdown]
# ## 1. One session: raw data
#
# Files are streamed from the DANDI S3 bucket with `remfile` and a local disk cache. Position is the
# midpoint of the two LEDs and head direction is the angle of the LED1 to LED2 vector. Only periods when
# the rat runs faster than 2.5 cm/s are used for tuning.
#
# Because the LED order relative to the nose is not documented, head direction is defined up to a
# constant offset. This affects preferred directions, not tuning strength.

# %%
sessions = list_sessions()
print(f"{len(sessions)} NWB files in DANDI:000582")
PATH = "sub-11265/sub-11265_ses-01020602_behavior+ecephys.nwb"
units, position, head_dir, moving, meta = load_session(dict(sessions)[PATH])
print(meta)
print(units)
print(position)

# %%
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
    axes[2].vlines(units[u].get(*win).t, k + 0.6, k + 1.4, lw=0.6)
axes[2].set_yticks(range(1, len(units) + 1), units.metadata["unit_name"].values, fontsize=7)
axes[2].set_xlabel("time (s)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. One session: what does each neuron encode?
#
# For each unit we compute:
#
# - a **rate map** (2.5 cm bins, Gaussian-smoothed) with `pynapple.compute_tuning_curves`;
# - its **spatial autocorrelogram** and **gridness score**: the correlation of the autocorrelogram with
#   itself rotated by 60 and 120 degrees minus the correlation at 30, 90 and 150 degrees. A hexagonal
#   lattice gives a high score;
# - a **head-direction tuning curve** (6 degree bins) and its **mean vector length (MVL)**, which is 0 for
#   untuned cells and 1 for perfectly tuned cells.

# %%
maps, occ = rate_maps(units, position, moving, meta["arena"])
curves, centers = hd_curves(units, head_dir, moving)
mvl, pref = mean_vector(curves, centers)
si = spatial_info(maps, occ)
acs = [autocorr2d(m) for m in maps]
gs = [gridness(a) for a in acs]
pd.DataFrame(dict(unit=units.metadata["unit_name"].values, spatial_info=si, gridness=gs, mvl=mvl)).round(2)

# %%
display(Image("figures/02_single_session_units.png"))

# %% [markdown]
# Several units in this layer III session (t1c2, t4c4, t8c1) have a clear hexagonal autocorrelogram, and
# most of these grid cells are also tuned to head direction. They are conjunctive cells. Other units
# (t2c1, t4c1) are tuned to heading but have no grid structure.
#
# ## 3. All sessions
#
# The same metrics are computed for every unit in all 118 sessions. For a null distribution, each spike
# train is circularly shifted in time by at least 20 s (100 shuffles per unit). This keeps the spike
# train's structure but removes its relation to behaviour. A unit counts as a grid cell or HD cell if its
# score exceeds the 99th percentile of the pooled shuffle distribution, as in Sargolini et al. (2006).
#
# Although every file's `session_description` says "1 x 1 meter enclosure", the tracking data in 22
# sessions span about 1.5 m or 2 m. The 2 m sessions have almost no occupancy in the corners, which
# suggests a circular arena. The arena size is therefore inferred per session from the tracking extent.
#
# This step takes about 10 minutes on 4 cores.

# %%
analysis = load_module("03_analyze_all_sessions.py")
rows, all_maps, all_acs, all_curves, shuf = [], [], [], [], []
with Pool(os.cpu_count()) as pool:
    for r, m, a, c, s in tqdm(pool.imap(analysis.analyze, sessions), total=len(sessions)):
        rows += r
        all_maps += list(m)
        all_acs += list(a)
        all_curves += list(c)
        shuf += s
df = pd.DataFrame(rows)
df.to_csv("results/units.csv", index=False)
obj = lambda x: np.array(x + [None], dtype=object)[:-1]
np.savez_compressed("results/maps.npz", maps=obj(all_maps), acs=obj(all_acs), curves=np.array(all_curves),
                    shuffles=np.array(shuf))
print(f"{len(df)} units from {df.path.nunique()} sessions and {df.subject.nunique()} rats")
df.groupby("layer").size()

# %% [markdown]
# ## 4. Population decoding
#
# If these cells tell the animal where it is and which way it faces, then the animal's heading and
# position should be readable from their spikes. For every session with at least 5 units, a Bayesian
# decoder (`pynapple.decode_bayes`, 1 s windows) is fit on even minutes and tested on odd minutes, and
# vice versa. Chance is the error obtained when decoding time-shifted spike trains with the same model.

# %%
decoding = load_module("04_decode_position_heading.py")
with Pool(os.cpu_count()) as pool:
    res = [r for r in tqdm(pool.imap(decoding.run, sessions), total=len(sessions)) if r is not None]
dec = pd.DataFrame([r[0] for r in res])
dec.to_csv("results/decoding.csv", index=False)
best = dec[dec.hd_err.notna()].sort_values("n_units", ascending=False).index[0]
np.savez("results/decoding_example.npz", path=dec.loc[best, "path"],
         **{f"{k}_{f}_{i}": v for k, folds_ in res[best][1].items() for f, fold in enumerate(folds_)
            for i, v in zip(("t", "dec", "true"), fold)})
dec.round(1)

# %% [markdown]
# ## 5. Results

# %%
# %run 05_visualize.py

# %%
display(Image("figures/03_population_metrics.png"))

# %% [markdown]
# Both gridness and head-direction MVL of the recorded units have long right tails that the shuffled data
# do not have. Grid cells and head-direction cells form separate clusters in the gridness vs MVL plane,
# with conjunctive cells where they overlap. Preferred directions of HD cells cover all directions, so
# the population can represent any heading.

# %%
display(Image("figures/04_classes_by_layer.png"))
pd.read_csv("results/class_by_layer.csv", index_col=0)

# %% [markdown]
# The layer distribution matches Sargolini et al. (2006). Pure grid cells are most common in layers II
# and III. Conjunctive grid x HD cells are concentrated in layer III. Pure head-direction cells dominate
# layers V and VI. Many layer II sessions lack the second LED, so their units cannot be tested for
# head-direction tuning (shown as "grid (no HD data)").

# %%
display(Image("figures/05_example_cells.png"))

# %%
display(Image("figures/06_decoding.png"))
print(open("results/summary.txt").read())

# %% [markdown]
# ## Interpretation
#
# **Which way am I facing?** Over half of the MEC units with heading data are tuned to head direction, and
# their preferred directions cover the full circle. From 5 to 19 simultaneously recorded cells, heading
# on held-out data is decoded with a median error of about 38 degrees (chance about 90 degrees), and the
# decoder beats chance in 38 of 42 sessions.
#
# **Where am I?** About a fifth of units are grid cells, with firing fields on a hexagonal lattice. The
# lattice is periodic, so a few cells from one grid module can tell apart nearby locations but leave
# distant locations ambiguous. That explains why position decoding beats chance in most sessions
# (48 of 55) but is coarse in absolute terms (median 44 cm vs 51 cm chance). An unambiguous position
# code is thought to need grid cells from several modules with different spacings, plus input from
# hippocampal place cells, which these recordings do not include.
#
# **How could the two be combined?** Conjunctive grid x HD cells, concentrated in layer III, fire only at
# grid locations and only for one heading. This is the type of signal a path-integration circuit needs to
# shift its position estimate in the direction of travel. The data are consistent with the MEC holding a
# self-motion-based map in which heading and position signals are combined, as proposed in the original
# study.
#
# **Caveats.** Recording sessions are short (10 to 20 minutes) and contain few units. Unit counts per
# layer are unequal. Head direction is defined up to an unknown constant offset. Arena size in 22 sessions
# was inferred from tracking data because the metadata state a 1 m box for all sessions.
