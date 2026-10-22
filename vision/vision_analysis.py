# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How does a mouse brain turn light into a picture of the world?
#
# This notebook follows visual signals from the thalamus into visual cortex using
# Neuropixels recordings from the Allen Brain Observatory, Visual Coding
# Neuropixels dataset ([DANDI:000021](https://dandiarchive.org/dandiset/000021)).
# Awake, head-fixed mice watched a fixed battery of visual stimuli while up to six
# probes recorded spikes from the visual thalamus (LGN, labelled `LGd`) and six
# visual cortical areas: primary visual cortex (V1, `VISp`) and the higher areas
# LM, RL, AL, PM and AM.
#
# We use 13 of the 32 sessions, those with at least 40 well-isolated units in both
# LGN and V1, and ask four questions, one per stimulus:
#
# 1. Timing (full-field flashes): in what order do areas respond?
# 2. Space (small Gabor patches on a grid): which part of the visual field does each neuron watch?
# 3. Features (drifting gratings): which areas care about edge orientation?
# 4. The picture (118 natural photographs): can we tell which photo the mouse saw from its neurons?
#
# Each step lives in its own script (`02_...py` to `05_...py`); this notebook runs
# them in order and shows their outputs.

# %% [markdown]
# ## Setup and data loading
#
# `common.py` streams each session's NWB file from DANDI with `remfile` + `h5py`,
# keeps only LGN and visual-cortex units that pass the Allen Institute's default
# quality thresholds (ISI violations < 0.5, amplitude cutoff < 0.1, presence
# ratio > 0.9), and caches spike times locally (about 1 minute per session the first
# time). Spikes become a pynapple `TsGroup` with each unit's area and probe as
# metadata; each stimulus table becomes a pynapple `IntervalSet`.
#
# Some probes stop recording for long stretches. Periods listed in the file's
# `invalid_times` table, plus any gap longer than 5 s with no spikes on a whole
# probe, are treated as invalid, and trials that overlap them are dropped for that
# probe's units.

# %%
import importlib

import pandas as pd
from IPython.display import Image, display

from common import AREAS, SESSIONS, load_session

step = {n: importlib.import_module(m) for n, m in [
    (1, "01_load_data"), (2, "02_flash_latency"), (3, "03_receptive_fields"),
    (4, "04_orientation_tuning"), (5, "05_natural_scene_decoding")]}
print(f"{len(SESSIONS)} sessions")

# %%
units, stims, invalid = load_session(step[1].SESSION)
step[1].summarize(units, stims, invalid)

# %% [markdown]
# ### Raw data
#
# Every quality-filtered unit in one session over 1.5 s, while six natural images
# were flashed back to back. Firing is sparse and ongoing; each image evokes a
# volley of spikes in V1 and other areas.

# %%
step[1].plot_raw(units, stims)
display(Image("figures/01_raw_raster.png"))

# %% [markdown]
# ## 1. Timing: signals flow from thalamus to V1 to higher areas
#
# For each session and area, all units are pooled into one population PSTH around
# 150 full-screen white or black flashes. The latency is when that population rate
# first reaches half its peak.

# %%
step[2].main()
display(Image("figures/02_flash_latency.png"))
pd.read_csv("results/flash_latency_summary.csv", index_col=0).round(1)

# %% [markdown]
# LGN responds first (median 38 ms after the flash), V1 next (44 ms), then LM, RL
# and AL (48-51 ms), and finally PM and AM (56-61 ms). LGN led V1 in 8 of the 12
# mice where both were recorded, with a median lead of 5.5 ms. This ordering
# matches the anatomical hierarchy: retina to LGN to V1 to higher visual areas.
# One V1 recording (session 799864342) responded weakly to flashes, which pushed
# its half-peak time to 92 ms.

# %% [markdown]
# ## 2. Space: each neuron sees one patch of the world
#
# Gabor patches 20 degrees across were flashed at 81 positions on a 9 x 9 grid.
# Averaging each unit's response per position gives its receptive-field (RF) map.

# %%
step[3].main()
display(Image("figures/03_receptive_fields.png"))
pd.read_csv("results/receptive_fields_summary.csv", index_col=0).round(2)

# %% [markdown]
# Most units (65% in LGN, 83% in V1, 69-82% in higher areas) respond to only part of
# the screen. Neighbouring V1 units share nearby RFs, and recordings from different
# mice cover different patches (bottom right), so the population as a whole tiles
# the visual field, much like pixels. RFs are smallest in V1 (median about
# 700 deg²) and larger in higher areas (1100-1400 deg²), so higher areas pool
# information over more of the scene. The grid spacing is 10 degrees and the
# patches are 20 degrees wide, so these sizes are coarse upper estimates. LGN's
# median looks large because many LGN maps are weak and broad under this measure.

# %% [markdown]
# ## 3. Features: orientation selectivity appears in cortex
#
# Full-screen gratings drifted in 8 directions at 5 speeds. For each unit we use its
# preferred speed and compute a global orientation selectivity index (OSI) from its
# mean firing rate in each direction.

# %%
step[4].main()
display(Image("figures/04_orientation_tuning.png"))
pd.read_csv("results/orientation_tuning_summary.csv", index_col=0).round(3)

# %% [markdown]
# LGN units barely care about orientation (median OSI 0.09; only 2% above 0.3). In V1
# the median OSI is 0.25 and 41% of units exceed 0.3, with some responding almost
# only to one orientation. Higher areas keep this property (medians 0.19-0.22).
# This is the classic transformation from spot-like thalamic inputs to
# edge-detecting cortical neurons, the first step in describing shapes.

# %% [markdown]
# ## 4. The picture: neural populations identify natural images
#
# Each of 118 photographs was flashed about 50 times. A Poisson naive-Bayes decoder
# learns each image's average population response from some repeats and names the
# image on held-out repeats. Chance is 1 in 118 (0.85%).

# %%
step[5].main()
display(Image("figures/05_natural_scene_decoding.png"))
pd.read_csv("results/decoding_vs_population_size.csv").groupby(["area", "n_units"])["accuracy"].mean() \
    .unstack().reindex(AREAS).round(3)

# %% [markdown]
# With only 30 units, V1 names the correct photo on a median of 30% of
# presentations, about 35 times chance. Every area does far better than chance.
# Pooling units across mice (possible because every mouse saw the same photos)
# shows accuracy rising steeply with population size: 300 LGN or V1 units identify
# the image on about 97% of single 250 ms glimpses. LM follows closely (94%), and
# AL, PM, RL and AM are lower (72-88%). One caveat is that pooling across mice
# ignores correlated noise between neurons recorded together, which can raise
# accuracy above what a single brain achieves.
#
# LGN carries nearly as much image-specific information as V1. Cortex does not
# add information about the image; it re-formats it. LGN describes the scene point
# by point, while V1 and later areas describe it with oriented edges in larger and
# larger patches of space, a format that makes shapes and objects easier to read
# out.

# %% [markdown]
# ## Summary
#
# Light hitting the retina reaches the LGN in under 40 ms and V1 about 5 ms later,
# then spreads through higher visual areas over the next 10-20 ms. Along the way:
#
# - each neuron reports on one patch of the visual field, and together they tile it;
# - V1 converts LGN's mostly orientation-blind signals into edge-orientation
#   detectors, and higher areas pool over larger regions;
# - a few hundred neurons, in thalamus or cortex, carry enough information to
#   tell which of 118 photographs the mouse is looking at from a single glimpse.
#
# The "picture of the world" is distributed across many neurons. No single neuron
# holds it, but the population code can be read out with a simple decoder.
