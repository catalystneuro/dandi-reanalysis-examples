# How does an animal know where it is located in space?

This analysis looks at the question in single neurons of the rat medial entorhinal cortex (MEC), using
real recordings from the DANDI Archive. It asks whether MEC neurons encode which way the animal faces
(head-direction cells) and where it is (grid cells), how those signals are spread across layers, and
whether heading and position can be decoded from population activity.

## Dataset

[DANDI:000582](https://dandiarchive.org/dandiset/000582/0.251111.2151), version 0.251111.2151:
"Conjunctive Representation of Position, Direction, and Velocity in Entorhinal Cortex"
(Sargolini et al., Science 2006, https://doi.org/10.1126/science.1125572).

- 118 NWB files, 15 rats, 620 sorted units from MEC layers II, III, V and VI.
- Tetrode spike times, plus position of one or two head-mounted LEDs sampled at 50 Hz during 10 to
  20 minute foraging sessions.
- Files are streamed from S3 with `remfile` and a local disk cache (about 16 MB each).

Data notes:

- Position values are stored in cm (the series `conversion` is 0.01 to metres).
- 47 of 118 sessions (167 units) have only one LED. They are used for spatial measures but not for
  head direction.
- Every file's `session_description` says "1 x 1 meter enclosure", but in 22 sessions the tracking
  data span about 1.5 m (2 sessions) or 2 m (20 sessions). The 2 m sessions have almost no corner
  occupancy, which suggests a circular arena. Arena size and centre are inferred per session from the
  tracking extent (`spatial_lib.load_session`).
- The LED order relative to the nose is not documented, so head direction is defined up to a constant
  offset. This affects preferred directions only.

## Approach

1. Position is the LED midpoint and head direction is the LED1 to LED2 angle. Only periods with
   running speed above 2.5 cm/s are used.
2. Rate maps (2.5 cm bins, smoothed), spatial information, spatial autocorrelograms, gridness scores,
   and head-direction tuning curves (6 degree bins, mean vector length) are computed with pynapple.
3. Significance: each spike train is circularly time-shifted 100 times (shift of at least 20 s). Grid
   and HD thresholds are the 99th percentile of the pooled shuffle distribution, following Sargolini
   et al. (2006).
4. Bayesian decoding (`pynapple.decode_bayes`, 1 s windows) of heading (12 degree bins) and position
   (10 cm bins), fit on even minutes and tested on odd minutes and vice versa, for sessions with at
   least 5 units. Chance is the decoding error for time-shifted spike trains.

## Results

Thresholds: gridness > 0.84, MVL > 0.27, spatial information > 1.18 bits/spike.

| Layer | Units | Pure grid | Grid (no HD data) | Conjunctive grid x HD | Pure HD | Other spatial | Unclassified |
|---|---|---|---|---|---|---|---|
| II | 71 | 9 | 10 | 0 | 2 | 12 | 38 |
| III | 227 | 30 | 5 | 36 | 50 | 14 | 92 |
| V | 118 | 1 | 6 | 6 | 64 | 0 | 41 |
| VI | 198 | 15 | 15 | 3 | 90 | 5 | 70 |

Six units with no layer label (all pure HD) are not in the table.

- 136 of 620 units (22%) are grid cells. 257 of the 453 units with heading data (57%) are HD-tuned.
- Grid cells are most common in layers II and III. Conjunctive grid x HD cells are mostly in layer III.
  Pure HD cells dominate layers V (54%) and VI (45%). This matches the layer pattern in the original paper.
- HD cell preferred directions cover the full circle.
- Heading decoding: median error 38 degrees vs 90 degrees chance. It beats the 5th percentile of
  chance in 38 of 42 sessions.
- Position decoding: median error 44 cm vs 51 cm chance. It beats chance in 48 of 55 sessions. The
  position code is weaker in absolute terms, as expected from a few periodic grid cells.

Interpretation: the MEC holds a heading signal that is precise enough to track the animal's direction
from a handful of cells, and a periodic position signal from grid cells. Conjunctive cells combine the
two, which is the input a path-integration circuit would need to update position from self-motion. A
unique position estimate likely needs more grid modules and hippocampal place cells, which are not
part of this dataset.

## Files

- `spatial_coding_mec.py` / `spatial_coding_mec.ipynb`: consolidated, executed notebook (jupytext percent format).
- `spatial_lib.py`: data loading and metric functions.
- `01_load_data.py`: inspect one NWB file.
- `02_single_session.py`: raw data and per-unit figures for one session.
- `03_analyze_all_sessions.py`: metrics and shuffles for all units (writes `results/units.csv` and `results/maps.npz`; the 44 MB maps file is not committed).
- `04_decode_position_heading.py`: population decoding (writes `results/decoding.csv`).
- `05_visualize.py`: classification, summary tables and figures.
- `figures/`: 01 raw data, 02 single-session units, 03 population metrics, 04 classes by layer,
  05 example cells, 06 decoding.
- `results/`: per-unit and per-session tables, `summary.txt`.

## Reproduce

```
pip install pynapple pynwb remfile h5py dandi matplotlib scipy tqdm jupytext nbconvert ipykernel
python 03_analyze_all_sessions.py && python 04_decode_position_heading.py && python 05_visualize.py
python 02_single_session.py
```

The full run takes about 15 minutes on 4 cores.
