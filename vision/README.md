# How does a mouse brain turn light into a picture of the world?

An analysis of the Allen Institute Visual Coding Neuropixels dataset,
[DANDI:000021](https://dandiarchive.org/dandiset/000021). It follows visual
responses from the visual thalamus (LGN) through primary visual cortex (V1) to five
higher visual areas (LM, RL, AL, PM, AM), using 13 sessions (13 mice) that each
contain at least 40 quality-filtered units in both LGN and V1.

Start with `vision_analysis.ipynb`, the executed notebook. Its source is
`vision_analysis.py` (jupytext percent format).

## Questions and findings

| Step | Stimulus | Question | Result |
|---|---|---|---|
| 1 | Full-field flashes | In what order do areas respond? | Median latency to half-peak: LGN 38 ms, V1 44 ms, LM/RL/AL 48-51 ms, PM/AM 56-61 ms |
| 2 | Gabor patches on a 9 x 9 grid | Which part of the visual field does a neuron watch? | 65-83% of units have a spatially restricted receptive field; V1 RFs are smallest (median about 700 deg²), higher areas about 1100-1400 deg² |
| 3 | Drifting gratings | Where does orientation selectivity appear? | Median OSI 0.09 in LGN vs 0.25 in V1 (2% vs 41% of units above 0.3); higher areas 0.19-0.22 |
| 4 | 118 natural photographs | Can the population tell which photo was shown? | 30 V1 units: 30% correct (chance 0.85%); 300 pooled LGN or V1 units: about 97% |

## Files

- `common.py`: streams NWB files from DANDI (remfile + h5py), applies unit quality
  filters, handles invalid periods, builds pynapple objects, and provides spike-count
  and PSTH helpers.
- `plotting.py`: shared colours and output folders.
- `01_load_data.py`: one-session summary and raw spike raster.
- `02_flash_latency.py`, `03_receptive_fields.py`, `04_orientation_tuning.py`,
  `05_natural_scene_decoding.py`: one step each. Every script runs on its own and
  writes to `figures/` and `results/`.
- `figures/`: one PNG per step.
- `results/`: per-unit and per-session tables (CSV) and summaries.

## Methods notes

- Units pass the Allen default quality thresholds: ISI violations < 0.5, amplitude
  cutoff < 0.1, presence ratio > 0.9. Area is the CCF structure of the unit's peak
  channel.
- Trials are dropped for a probe's units when they overlap that probe's entries in
  the NWB `invalid_times` table, or any gap longer than 5 s with no spikes on the
  whole probe. One probe (session 743475441, V1) has no spikes for the first
  2136 s, and that period is not listed in `invalid_times`.
- Latency is measured on per-area population PSTHs, not single units, because
  single-unit onset estimates from 150 flashes were too noisy.
- OSI is the global OSI on raw mean rates at the preferred temporal frequency.
- Decoding uses a Poisson naive-Bayes classifier with 5-fold cross-validation over
  repeats. The pooled "pseudo-population" curve combines mice by matching repeat
  number, which ignores noise correlations, so it is an upper estimate for a
  single brain.
- Stimulus image templates are not included in these NWB files, so the
  photographs themselves are not shown.

## Reproducing

```
pip install pynapple pynwb remfile h5py matplotlib scipy pandas tqdm jupytext nbconvert ipykernel
cd vision
python 01_load_data.py   # or run any step script directly
jupytext --to ipynb vision_analysis.py
jupyter nbconvert --to notebook --execute --inplace vision_analysis.ipynb
```

The first run downloads the selected units of each session (about 1 minute and
100-200 MB per session) to `~/.cache/vision_e` (override with `VISION_CACHE`).
Versions used: pynapple 0.11.4, pynwb 4.2.0, Python 3.13.
