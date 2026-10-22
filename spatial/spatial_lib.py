"""Shared helpers: load a DANDI:000582 session and compute spatial / directional tuning metrics."""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter, rotate
from dandi.dandiapi import DandiAPIClient

DANDISET, VERSION = "000582", "0.251111.2151"
CACHE = "/tmp/remfile_cache"

BOX_SIDES = np.array([100.0, 150.0, 200.0])  # cm; box sizes found in the tracking data
BIN_CM = 2.5
SPEED_MIN = 2.5  # cm/s
HD_BINS = 60  # 6 deg bins
N_SHUFFLE = 100
MIN_SHIFT = 20.0  # s


def list_sessions():
    with DandiAPIClient() as c:
        ds = c.get_dandiset(DANDISET, VERSION)
        return sorted(
            (a.path, a.get_content_url(follow_redirects=1, strip_query=True))
            for a in ds.get_assets()
            if a.path.endswith(".nwb")
        )


def open_nwb(url):
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read(), io


def load_session(url):
    """Return units (TsGroup), position (TsdFrame, cm), head direction (Tsd, rad), moving epochs, metadata."""
    nwbfile, io = open_nwb(url)
    pos_mod = nwbfile.processing["behavior"]["Position"]
    led1 = pos_mod["SpatialSeriesLED1"]
    t = led1.timestamps[:]
    # raw values are in cm (conversion factor 0.01 to metres)
    p1 = led1.data[:]
    if "SpatialSeriesLED2" in pos_mod.spatial_series:
        p2 = pos_mod["SpatialSeriesLED2"].data[:]
        ok = np.all(np.isfinite(p1), 1) & np.all(np.isfinite(p2), 1)
        t, p1, p2 = t[ok], p1[ok], p2[ok]
        xy = (p1 + p2) / 2
        hd = np.mod(np.arctan2(p1[:, 1] - p2[:, 1], p1[:, 0] - p2[:, 0]), 2 * np.pi)
        head_dir = nap.Tsd(t=t, d=hd)
    else:  # single-LED sessions: position only, no head direction
        ok = np.all(np.isfinite(p1), 1)
        t, xy = t[ok], p1[ok]
        head_dir = None

    position = nap.TsdFrame(t=t, d=xy, columns=["x", "y"])

    # Every file's session_description says "1 x 1 meter enclosure", but some sessions' tracking spans
    # ~1.5 m or ~2 m. Infer the box from the tracking extent: nearest standard side, centred on the data.
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    side = BOX_SIDES[np.argmin(np.abs(BOX_SIDES - (hi - lo).max()))]
    centre = (lo + hi) / 2
    arena = [(c - side / 2, c + side / 2) for c in centre]

    # speed from lightly smoothed position
    fs = 1 / np.median(np.diff(t))
    xs = gaussian_filter(xy, sigma=(0.2 * fs, 0), mode="nearest")
    speed = np.r_[0, np.linalg.norm(np.diff(xs, axis=0), axis=1) * fs]
    moving = nap.Tsd(t=t, d=speed).threshold(SPEED_MIN).time_support

    udf = nwbfile.units.to_dataframe()
    units = nap.TsGroup(
        {i: nap.Ts(np.asarray(st)) for i, st in zip(udf.index, udf["spike_times"])},
        unit_name=udf["unit_name"].values,
        histology=udf["histology"].values,
    )
    meta = dict(session_id=nwbfile.session_id, subject_id=nwbfile.subject.subject_id, duration=t[-1] - t[0], fs=fs,
                arena=arena, box_side=side)
    io.close()
    return units, position, head_dir, moving, meta


def smooth_map(rate, occ, sigma=1.5):
    """NaN-aware Gaussian smoothing of a rate map; unvisited bins stay NaN."""
    valid = occ > 0
    r = np.where(valid, rate, 0.0)
    num = gaussian_filter(r * valid, sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    out = num / np.maximum(den, 1e-12)
    out[~valid] = np.nan
    return out


def rate_maps(units, position, epochs, arena):
    nb = int(round((arena[0][1] - arena[0][0]) / BIN_CM))
    tc = nap.compute_tuning_curves(units, position, bins=nb, range=arena, epochs=epochs, fs=50.0)
    occ = tc.attrs["occupancy"]
    maps = np.stack([smooth_map(np.nan_to_num(tc.sel(unit=u).values), occ) for u in tc.coords["unit"].values])
    return maps, occ


def hd_curves(units, head_dir, epochs):
    tc = nap.compute_tuning_curves(units, head_dir, bins=HD_BINS, range=[(0, 2 * np.pi)], epochs=epochs, fs=50.0)
    vals = tc.values  # (unit, bin)
    # circular smoothing (1 bin)
    vals = np.stack([gaussian_filter(np.nan_to_num(v), 1, mode="wrap") for v in vals])
    centers = tc.coords[tc.dims[1]].values
    return vals, centers


def mean_vector(curves, centers):
    z = (curves * np.exp(1j * centers)).sum(1) / np.maximum(curves.sum(1), 1e-12)
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)


def spatial_info(maps, occ):
    p = occ / occ.sum()
    out = []
    for m in maps:
        m = np.nan_to_num(m)
        lam = (p * m).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            term = p * (m / lam) * np.log2(m / lam)
        out.append(np.nansum(term))
    return np.array(out)


def autocorr2d(m):
    """Pearson spatial autocorrelogram of a rate map with NaNs (Sargolini et al. 2006)."""
    v = np.isfinite(m)
    x = np.where(v, m, 0.0)
    n = v.shape[0]
    s = 2 * n - 1

    def xc(a, b):
        return np.real(np.fft.ifft2(np.fft.fft2(a, (s, s)) * np.conj(np.fft.fft2(b, (s, s)))))

    N = xc(v.astype(float), v.astype(float))
    sx = xc(x, v.astype(float))
    sy = xc(v.astype(float), x)
    sxx = xc(x * x, v.astype(float))
    syy = xc(v.astype(float), x * x)
    sxy = xc(x, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (N * sxy - sx * sy) / (np.sqrt(N * sxx - sx**2) * np.sqrt(N * syy - sy**2))
    r[N < 20] = np.nan
    return np.fft.fftshift(r)


def gridness(ac):
    """Gridness score: min(r60, r120) - max(r30, r90, r150) on an annulus around the central peak.

    The inner radius is the edge of the central field: the first radius where the radial
    autocorrelation drops below 0 or reaches a local minimum. The outer radius is swept
    and the maximum score is kept.
    """
    c = np.array(ac.shape) // 2
    yy, xx = np.indices(ac.shape)
    rr = np.hypot(yy - c[0], xx - c[1])
    a0 = np.nan_to_num(ac)
    radial = np.array([np.nanmean(ac[(rr >= r) & (rr < r + 1)]) for r in range(c[0])])
    d = np.diff(radial)
    edge = np.where((radial[1:] < 0) | (np.r_[d[1:], 1] > 0) & (d < 0))[0]
    if len(edge) == 0:
        return np.nan
    r_in = max(edge[0] + 1, 2)
    if r_in + 4 >= int(c[0] * 0.9):
        return np.nan
    rots = {a: rotate(a0, a, reshape=False, order=1) for a in (30, 60, 90, 120, 150)}
    best = -np.inf
    for r_out in range(r_in + 4, int(c[0] * 0.9)):
        ring = (rr >= r_in) & (rr <= r_out) & np.isfinite(ac)
        cors = {a: np.corrcoef(a0[ring], rots[a][ring])[0, 1] for a in rots}
        g = min(cors[60], cors[120]) - max(cors[30], cors[90], cors[150])
        best = max(best, g)
    return best


def shuffled_groups(spikes, start, end, rng):
    """N_SHUFFLE circularly time-shifted copies of one spike train (shift >= MIN_SHIFT s)."""
    T = end - start
    shifts = rng.uniform(MIN_SHIFT, T - MIN_SHIFT, N_SHUFFLE)
    st = spikes.t - start
    return nap.TsGroup({k: nap.Ts(np.sort(np.mod(st + s, T)) + start) for k, s in enumerate(shifts)})
