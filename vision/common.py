"""Shared helpers: stream one Allen Visual Coding Neuropixels session (DANDI 000021)
and turn it into pynapple objects.

Only units in the visual thalamus (LGd) and visual cortex (VIS*) are read, and only
units that pass the Allen Institute's default quality thresholds. Reading the whole
units table through pynwb would pull ~1 GB of spike times per session, so spike
times are sliced per unit with h5py over a remfile stream and cached locally.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pynapple as nap
import remfile

DANDISET = "000021"
CACHE_DIR = Path(os.environ.get("VISION_CACHE", Path.home() / ".cache" / "vision_e"))

# Visual areas in anatomical hierarchy order (Harris et al. 2019, Siegle et al. 2021)
AREAS = ["LGd", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
AREA_LABELS = {
    "LGd": "LGN (thalamus)",
    "VISp": "V1",
    "VISl": "LM",
    "VISrl": "RL",
    "VISal": "AL",
    "VISpm": "PM",
    "VISam": "AM",
}

# Sessions with at least 40 good units in both LGd and VISp (from a survey of all 32
# session files in DANDI 000021). Keys are session ids, values are DANDI asset ids.
SESSIONS = {
    "715093703": "58703c97-c0a9-4736-b684-73c85c1a444a",
    "719161530": "02291b99-e583-498b-9929-b68bba2c50e2",
    "743475441": "522e8054-34ca-4579-ae80-350d0b24e0f4",
    "750749662": "3876c5f1-f38a-4c89-8f54-6128538f0066",
    "751348571": "be9f8fd8-8f16-4a66-acc6-9e04697650f3",
    "754829445": "488a9bea-96fb-4028-a33a-08650f50be63",
    "755434585": "edf10182-5a4c-454f-ad23-47987a5ca256",
    "756029989": "e0037b53-6d0e-4756-8ca4-f0dcb51ad339",
    "757970808": "dfc3db15-066a-4a07-b615-a4d7e85c44e1",
    "760345702": "47634abd-db85-48f5-9c33-01887a59d3bc",
    "761418226": "cdeec90c-9ebe-431c-b3f7-b9f7d93ff37f",
    "763673393": "8a17b967-2aa9-4d6a-812c-92b62cf799d7",
    "799864342": "264f3b24-10d9-4100-98cf-d40b6c2a2068",
}

SILENT_GAP = 5.0  # s without a single spike on a whole probe

STIM_COLUMNS = {
    "flashes_presentations": ["color"],
    "gabors_presentations": ["x_position", "y_position", "orientation"],
    "drifting_gratings_presentations": ["orientation", "temporal_frequency"],
    "natural_scenes_presentations": ["frame"],
}


def asset_url(asset_id):
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def _download(session_id, path):
    f = h5py.File(remfile.File(asset_url(SESSIONS[session_id])), "r")
    elec = f["general/extracellular_ephys/electrodes"]
    loc = dict(zip(elec["id"][:], elec["location"][:].astype(str)))
    depth = dict(zip(elec["id"][:], elec["probe_vertical_position"][:]))
    probe = dict(zip(elec["id"][:], elec["group_name"][:].astype(str)))
    u = f["units"]
    peak = u["peak_channel_id"][:]
    area = np.array([loc[p] for p in peak])
    good = (
        (u["isi_violations"][:] < 0.5)
        & (u["amplitude_cutoff"][:] < 0.1)
        & (u["presence_ratio"][:] > 0.9)
        & np.isin(area, AREAS)
    )
    idx = np.flatnonzero(good)
    ends = u["spike_times_index"][:]
    starts = np.concatenate([[0], ends[:-1]])
    st = u["spike_times"]
    out = {
        "unit_id": u["id"][:][idx],
        "area": area[idx],
        "depth": np.array([depth[p] for p in peak[idx]]),
        "waveform_duration": u["waveform_duration"][:][idx],
        "snr": u["snr"][:][idx],
        "probe": np.array([probe[p] for p in peak[idx]]),
    }
    spikes = [st[starts[i]:ends[i]] for i in idx]
    out["spike_counts"] = np.array([len(s) for s in spikes])
    out["spike_times"] = np.concatenate(spikes)
    for name, cols in STIM_COLUMNS.items():
        g = f["intervals"][name]
        out[f"{name}/start_time"] = g["start_time"][:]
        out[f"{name}/stop_time"] = g["stop_time"][:]
        for c in cols:
            v = g[c][:]
            if v.dtype.kind in "OS":
                v = v.astype(str).astype(float)
            out[f"{name}/{c}"] = v
    out.update(_invalid_times(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def _invalid_times(f):
    """Periods the Allen Institute marked as invalid, per probe or ("all") for the whole session."""
    if "invalid_times" not in f["intervals"]:
        return {"invalid/start_time": np.array([]), "invalid/stop_time": np.array([]), "invalid/probe": np.array([], dtype=str)}
    g = f["intervals/invalid_times"]
    tags = g["tags"][:].astype(str)
    tag_end = g["tags_index"][:]
    tag_start = np.concatenate([[0], tag_end[:-1]])
    probe = []
    for a, b in zip(tag_start, tag_end):
        p = [t for t in tags[a:b] if t.startswith("probe")]
        probe.append(p[0] if p else "all")
    return {
        "invalid/start_time": g["start_time"][:],
        "invalid/stop_time": g["stop_time"][:],
        "invalid/probe": np.array(probe),
    }


def load_session(session_id):
    """Return (units TsGroup, dict of stimulus IntervalSets, invalid periods).

    ``invalid`` maps each probe name to an IntervalSet of periods the Allen Institute
    marked as invalid for that probe, so analyses can drop the affected trials.
    """
    path = CACHE_DIR / f"ses-{session_id}.npz"
    if not path.exists():
        _download(session_id, path)
    d = np.load(path)
    splits = np.split(d["spike_times"], np.cumsum(d["spike_counts"])[:-1])
    units = nap.TsGroup(
        {int(uid): nap.Ts(t) for uid, t in zip(d["unit_id"], splits)},
        metadata={
            "area": d["area"],
            "depth": d["depth"],
            "waveform_duration": d["waveform_duration"],
            "snr": d["snr"],
            "probe": d["probe"],
        },
    )
    invalid = {}
    for p in np.unique(d["probe"]):
        flagged = np.isin(d["invalid/probe"], [p, "all"])
        start = list(d["invalid/start_time"][flagged])
        end = list(d["invalid/stop_time"][flagged])
        # Some probes are silent for long stretches that are not flagged in
        # invalid_times (e.g. no spikes for the first 2000 s). Treat any gap
        # longer than SILENT_GAP in the probe's pooled spike train as invalid.
        t = np.sort(np.concatenate([units[k].t for k in np.asarray(units.keys())[d["probe"] == p]]))
        edges = np.concatenate([[0.0], t, [d["spike_times"].max()]])
        gap = np.flatnonzero(np.diff(edges) > SILENT_GAP)
        start += list(edges[gap])
        end += list(edges[gap + 1])
        invalid[p] = nap.IntervalSet(start, end)
    stims = {}
    for name, cols in STIM_COLUMNS.items():
        stims[name.replace("_presentations", "")] = nap.IntervalSet(
            d[f"{name}/start_time"],
            d[f"{name}/stop_time"],
            metadata={c: d[f"{name}/{c}"] for c in cols},
        )
    return units, stims, invalid


def valid_trials(units, invalid, starts, window):
    """Boolean (n_trials, n_units): False where the window overlaps the unit's probe invalid times."""
    lo, hi = starts + window[0], starts + window[1]
    probes = units.metadata["probe"].values
    ok = {}
    for p, inv in invalid.items():
        bad = np.zeros(len(starts), bool)
        for a, b in zip(inv.start, inv.end):
            bad |= (lo < b) & (hi > a)
        ok[p] = ~bad
    return np.stack([ok[p] for p in probes], axis=1)


def counts_matrix(units, invalid, starts, window):
    """Spike counts, shape (n_trials, n_units), in [start + window[0], start + window[1]).

    Trials that overlap the unit's probe invalid times are NaN. Uses searchsorted on
    each unit's spike times so that windows may overlap (a pynapple IntervalSet
    would merge overlapping windows).
    """
    lo, hi = starts + window[0], starts + window[1]
    out = np.empty((len(starts), len(units)))
    for j, uid in enumerate(units.keys()):
        t = units[uid].t
        out[:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    out[~valid_trials(units, invalid, starts, window)] = np.nan
    return out


def psth(units, invalid, starts, window=(-0.1, 0.35), bin_size=0.001):
    """Trial-averaged firing rate (Hz), shape (n_bins, n_units), invalid trials excluded."""
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    ok = valid_trials(units, invalid, starts, window)
    out = np.full((len(edges) - 1, len(units)), np.nan)
    for j, uid in enumerate(units.keys()):
        s = starts[ok[:, j]]
        if len(s) == 0:
            continue
        t = units[uid].t
        i0 = np.searchsorted(t, s + window[0])
        i1 = np.searchsorted(t, s + window[1])
        rel = np.concatenate([t[a:b] - x for a, b, x in zip(i0, i1, s)])
        out[:, j] = np.histogram(rel, edges)[0] / len(s) / bin_size
    return edges[:-1] + bin_size / 2, out
