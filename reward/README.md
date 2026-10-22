# How does an animal learn what comes before a reward?

Mice learn that a tone predicts sucrose: they start licking during the tone before the sucrose arrives, and dopamine in the nucleus accumbens starts responding to the tone. This analysis follows that learning trial by trial in three cohorts that differed only in the time between cue-sucrose pairings. The number of pairings a mouse needed fell steeply as the pairings were spaced further apart.

## Data

[DANDI:001632](https://dandiarchive.org/dandiset/001632/draft) (draft version), "Duration between rewards controls the rate of behavioral and dopaminergic learning". Paper: Burke et al., Nature Neuroscience 2026, [doi:10.1038/s41593-026-02206-2](https://doi.org/10.1038/s41593-026-02206-2).

Head-fixed C57BL/6J mice heard a 250 ms, 12 kHz tone (CS+). Sucrose was delivered 1.25 s after tone onset. Dopamine release in the NAc was recorded with dLight1.3b by fiber photometry. The three dopamine cohorts used here:

| Cohort | Mean ITI | Mice | Pairings per session | Total pairings |
|---|---|---|---|---|
| `60sD` | 60 s | 6 | 50 | 400 to 449 |
| `600sD` | 600 s | 7 | 6 | 47 to 48 |
| `3600sD` | 3600 s | 5 | 2 | 16 to 18 |

148 sessions in total. Data read from each NWB file:

- `acquisition/eventLog`: `event_code` 15 = CS+ onset, 10 = sucrose solenoid, 5 = lick onset on the sucrose spout, with `timestamp`
- `acquisition/task_parameters`: `meanITI`, `CS_t_fxd` (cue-to-sucrose delay)
- `processing/photometry/photometry_dff`: `data` and `timestamps`

PyNWB 4.2 fails to read these files because the cached ndx-events 0.4.0 `MeaningsTable` spec conflicts with types now in NWB core, so the HDF5 paths above are read with h5py. Each file (0.3 to 14 MB) is fetched in one HTTP request.

## Approach

Per trial:

- anticipatory licking: lick rate in the 1.25 s cue-to-sucrose delay minus the rate in the 1.25 s before the cue
- dopamine cue response: mean dF/F 0 to 0.5 s after cue onset minus the 1 s pre-cue mean
- dopamine reward response: mean dF/F 0 to 0.5 s after sucrose minus the same baseline

Learning trial per mouse: the elbow of the cumulative curve of each measure (where the cumulative sum lies furthest below the line joining its start and end). A second criterion, the first trial where a 5-trial running mean reaches half the mouse's final level, is reported as a check. Perievent alignment uses pynapple.

## Results

Median pairings and hours of conditioning to learn (elbow criterion):

| ITI | Pairings (licking) | Pairings (dopamine) | Hours (licking) | Hours (dopamine) |
|---|---|---|---|---|
| 60 s | 168 | 126 | 2.8 | 2.1 |
| 600 s | 18 | 19 | 3.0 | 3.2 |
| 3600 s | 9 | 9 | 9 | 9 |

- In every mouse the dopamine cue response grew over training, and 16 of 18 mice developed clear anticipatory licking. The dopamine response to sucrose shrank as the cue response grew (`figures/04_peth_by_session.png`).
- From 60 s to 600 s ITI, the pairings needed fell about ninefold (licking) and sevenfold (dopamine) while the hours needed stayed near 2 to 3. Learning per pairing grew roughly in proportion to the time between pairings, consistent with the paper.
- Across all three cohorts, the log-log slope of pairings-to-learn against ITI was -0.63 (95% CI -0.83 to -0.43) for licking and -0.70 (-0.89 to -0.51) for dopamine. A slope of -1 would be exact proportionality. The 3600 s mice learned in the fewest pairings but over more hours.

Caveats:

- The 3600 s cohort had only 16 to 18 pairings, so its learning trial is coarsely resolved. By the half-asymptote criterion its dopamine cue response reached half its final size within 1 to 3 pairings.
- Several 3600 s mice showed a dopamine response to the tone on the first trial, before any pairing (mean 5.1 dF/F units versus about 0 in the other cohorts), likely a novelty response to a tone after an hour of silence.
- Two 60 s mice (F9, F11) developed little anticipatory licking, so their licking learning trials are not meaningful. They are kept in all statistics.
- With the half-asymptote criterion the licking relationship is weaker and not significant (slope -0.37, p = 0.09). The dopamine relationship holds under both criteria (slope -0.67, p < 0.001).
- Session 60sD-F8 day02a (recorded before a software crash) logged no licks while photometry was normal; its licking is treated as missing. One 3600 s session has a 15-minute NaN stretch in dF/F away from any cue, and one 600 s session has a single dropped photometry sample (traces are resampled to a uniform grid).
- These learning-trial definitions are simple and may differ from the paper's. The authors' code: https://github.com/namboodirilab/duration_between_rewards_controls_learning_rate

## Files

- `cue_reward_learning.py` / `cue_reward_learning.ipynb`: narrated notebook (jupytext percent format) that runs the full pipeline
- `01_load_data.py`: fetch sessions from DANDI into `data/*.npz` (not committed)
- `02_preprocess.py`: quality checks, raw-data figure, session table
- `03_analyze_cue_reward_learning.py`: per-trial measures and learning trials (`data/trials.csv`, `data/learning.csv`)
- `04_visualize.py`: summary figures
- `figures/`: all figures

## Reproduce

```
pip install pynapple dandi h5py requests numpy pandas scipy matplotlib tqdm jupytext nbconvert ipykernel
cd reward
python 01_load_data.py && python 02_preprocess.py && python 03_analyze_cue_reward_learning.py && python 04_visualize.py
# or
jupyter nbconvert --to notebook --execute cue_reward_learning.ipynb
```

Tested with Python 3.13, pynapple 0.11.4. Downloading all sessions takes a few minutes.
