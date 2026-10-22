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
# # How does an animal learn what comes before a reward?
#
# A mouse hears a short tone, and 1.25 s later a drop of sucrose arrives. At first the
# tone means nothing. Over many pairings the mouse starts licking during the tone,
# before the sucrose comes, and dopamine in the nucleus accumbens (NAc) starts
# responding to the tone itself. Both signals show that the animal has learned that
# the tone predicts reward.
#
# This notebook follows that learning trial by trial in real data from
# [DANDI:001632](https://dandiarchive.org/dandiset/001632/draft) (Burke et al., 2026,
# Nature Neuroscience, [doi:10.1038/s41593-026-02206-2](https://doi.org/10.1038/s41593-026-02206-2)).
# Head-fixed mice received cue-sucrose pairings while dopamine release in the NAc was
# recorded with the dLight1.3b sensor by fiber photometry. Three cohorts differed only
# in the mean time between pairings (inter-trial interval, ITI): 60 s, 600 s or 3600 s.
# Each session lasted about an hour (two hours for 3600 s), so the cohorts received
# 50, 6 or 2 pairings per day.
#
# The question here: does learning depend on how many pairings an animal experiences,
# or on how they are spaced in time?
#
# The pipeline is split into four scripts that this notebook runs in order:
#
# 1. `01_load_data.py`: fetch every session of the three dopamine cohorts from DANDI
# 2. `02_preprocess.py`: quality checks and a raw-data view
# 3. `03_analyze_cue_reward_learning.py`: per-trial licking and dopamine measures, learning trial per mouse
# 4. `04_visualize.py`: summary figures

# %%
import importlib
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

HERE = Path.cwd()
DATA = HERE / "data"
FIG = HERE / "figures"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 1. Load the data
#
# Each NWB file is small (0.3 to 14 MB), so it is fetched in a single HTTP request and
# read with h5py. Files are cached as `data/*.npz`, so re-running skips the download.
# Event codes in `acquisition/eventLog`: 15 = CS+ tone onset, 10 = sucrose solenoid,
# 5 = lick onset on the sucrose spout. Dopamine: `processing/photometry/photometry_dff`.

# %%
importlib.import_module("01_load_data").main()

# %% [markdown]
# ## 2. Quality checks and raw data
#
# All 148 sessions (18 mice) have one sucrose delivery 1.25 s after every tone.
# Photometry is sampled at about 120 Hz (130 Hz for the 3600 s cohort). One session
# has a 15-minute NaN stretch away from any cue, one has a single dropped sample,
# and in one session (60sD-F8 day02a, recorded before a software crash) no licks
# were logged. Its licking is treated as missing.

# %%
importlib.import_module("02_preprocess").main()
sessions = pd.read_csv(DATA / "sessions.csv")
sessions.groupby(["group", "subject"])["n_cs"].agg(sessions="count", pairings="sum")

# %% [markdown]
# The first five pairings for a naive 60 s ITI mouse. Dopamine rises after the
# sucrose and the mouse licks only once the sucrose is there. The tone itself
# triggers little.

# %%
display(Image(FIG / "01_raw_session.png"))
display(Image(FIG / "02_qc.png"))

# %% [markdown]
# ## 3. Measure learning on every trial
#
# For each tone presentation:
#
# - anticipatory licking: lick rate during the 1.25 s tone-to-sucrose delay minus the
#   rate in the 1.25 s before the tone
# - dopamine cue response: mean dF/F 0 to 0.5 s after tone onset minus the 1 s pre-tone baseline
# - dopamine reward response: mean dF/F 0 to 0.5 s after sucrose minus the same baseline
#
# The trial where each mouse learns is taken as the elbow of the cumulative curve of
# each measure: the point where the running sum bends upward. A second criterion
# (5-trial running mean reaches half the mouse's final level) is computed as a check.

# %%
importlib.import_module("03_analyze_cue_reward_learning").main()
learning = pd.read_csv(DATA / "learning.csv")
learning.round(2)

# %% [markdown]
# ## 4. What learning looks like
#
# One mouse per cohort (the one with the median dopamine learning trial). In every
# cohort the tone-evoked dopamine response (left band at 0 s) grows from nothing,
# and licking spreads backward from the sucrose into the delay period.
# The cyan dotted line marks the learning trial.

# %%
importlib.import_module("04_visualize").main()
display(Image(FIG / "03_example_heatmaps.png"))

# %% [markdown]
# Averaged per session, the dopamine response moves from the sucrose to the tone.
# The cue response grows while the sucrose response shrinks, which is the pattern
# expected when the reward becomes predicted. Anticipatory licking between 0 and
# 1.25 s grows along with it.

# %%
display(Image(FIG / "04_peth_by_session.png"))

# %% [markdown]
# ## 5. Pairings or time?
#
# Plotted against the number of pairings (left), the three cohorts learn at very
# different points: the 60 s ITI mice need hundreds of pairings, the 600 s ITI
# mice tens, the 3600 s ITI mice fewer than ten. Plotted against hours of
# conditioning (right; pairings times ITI), the 60 s and 600 s curves overlap.
# The 3600 s mice learn in the fewest pairings but over more hours.

# %%
display(Image(FIG / "05_learning_curves.png"))

# %%
summary = learning.groupby("group")[
    ["learned_trial_lick", "learned_trial_da", "learned_time_lick_h", "learned_time_da_h"]
].median().loc[["60sD", "600sD", "3600sD"]]
summary.columns = ["pairings to learn (licking)", "pairings to learn (dopamine)",
                   "hours to learn (licking)", "hours to learn (dopamine)"]
summary.round(1)

# %%
display(Image(FIG / "06_learned_trial_vs_iti.png"))

# %% [markdown]
# ## Results
#
# What the animal learns. In all 18 mice the dopamine response to the tone grew over
# training (it was absent at the start in the 60 s and 600 s cohorts), and 16 of 18
# mice developed clear licking during the tone. As the cue response grows, the dopamine response to the
# sucrose itself shrinks (figure 4). The reward-evoked signal transfers to the event
# that predicts it.
#
# How fast depends on the spacing between rewards. Using the elbow criterion,
# median pairings to learn were:
#
# | ITI | licking | dopamine | hours (licking) | hours (dopamine) |
# |---|---|---|---|---|
# | 60 s (6 mice) | 168 | 126 | 2.8 | 2.1 |
# | 600 s (7 mice) | 18 | 19 | 3.0 | 3.2 |
# | 3600 s (5 mice) | 9 | 9 | 9 | 9 |
#
# Going from 60 s to 600 s (ten times longer between rewards) cut the number of
# pairings needed by about ninefold for licking and sevenfold for dopamine, while
# the conditioning time needed stayed at about 2 to 3 hours. Learning per pairing
# grew roughly in proportion to the time between pairings, consistent with the
# source paper's conclusion.
#
# Across all three cohorts, the log-log slope of pairings-to-learn against ITI was
# -0.63 (95% CI -0.83 to -0.43) for licking and -0.70 (-0.89 to -0.51) for dopamine.
# A slope of -1 would mean exact proportionality. The 3600 s cohort pulls the slope
# toward zero: those mice learned in fewer pairings but over more hours.
#
# Caveats:
#
# - The 3600 s mice got only 2 pairings a day and 16 to 18 in total, so the learning
#   trial is poorly resolved at the low end. By the half-asymptote criterion their
#   dopamine cue response reached half its final size within 1 to 3 pairings.
# - Several 3600 s mice already showed a dopamine response to the tone on the very
#   first trial (mean 5.1 dF/F units versus about 0 in the other cohorts), before
#   any pairing. A tone after an hour of silence likely evokes a novelty response,
#   which confounds the early dopamine measure for that cohort.
# - Two 60 s mice (F9, F11) developed little anticipatory licking (at most 0.6 Hz
#   after the elbow), so their licking learning trials are not meaningful. They are
#   kept in all statistics.
# - The half-asymptote criterion gives a weaker, non-significant licking relationship
#   (slope -0.37, p = 0.09), mainly because the two weak-licking 60 s mice cross half
#   of their low final level on trial 1. The dopamine relationship holds under both
#   criteria (slope -0.67, p < 0.001).
# - The learning-trial definitions here are simple and may differ from the paper's.
#   The paper's own analysis code is at
#   https://github.com/namboodirilab/duration_between_rewards_controls_learning_rate.
