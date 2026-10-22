"""Load behavior events and NAc dopamine (dLight1.3b) photometry from DANDI:001632.

Each NWB file is small (0.3-14 MB), so it is fetched into memory with one HTTP
request and read with h5py. Streaming with remfile also works but issues many small
range requests (~30 s per session here). PyNWB 4.2 cannot construct these files
because the cached ndx-events 0.4.0 spec (MeaningsTable) conflicts with the event
types now in NWB core, so the HDF5 groups are read directly. The paths used here
are listed in the README.

Output: data/<subject>_<session>.npz, one per session, holding CS onset, sucrose
delivery and lick onset times plus the dF/F trace (float32), and data/index.json.
"""

import io
import json
from pathlib import Path

import h5py
import numpy as np
import requests
from dandi.dandiapi import DandiAPIClient
from tqdm import tqdm

DANDISET = "001632"
GROUPS = ["60sD", "600sD", "3600sD"]  # dopamine cohorts; the number is the mean ITI in s
OUT = Path(__file__).parent / "data"

CODE_CS = 15  # "sound 1 (CS+ onset)"
CODE_REWARD = 10  # "Fixed solenoid 3 (sucrose delivery)"
CODE_LICK = 5  # "Lick3 onset" (task parameter golicktube = 3 for the CS+)


def group_of(path):
    subject = path.split("/")[0].removeprefix("sub-")
    return subject.rsplit("-", 1)[0]


def read_session(url):
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    f = h5py.File(io.BytesIO(response.content), "r")
    params = dict(
        zip(
            [x.decode() for x in f["acquisition/task_parameters/param_name"][:]],
            [x.decode() for x in f["acquisition/task_parameters/param_value"][:]],
        )
    )
    ev = f["acquisition/eventLog"]
    codes = ev["event_code"][:]
    t = ev["timestamp"][:]
    dff = f["processing/photometry/photometry_dff"]
    out = dict(
        cs=t[codes == CODE_CS],
        reward=t[codes == CODE_REWARD],
        lick=t[codes == CODE_LICK],
        dff_t=dff["timestamps"][:],
        dff=dff["data"][:].astype(np.float32),
        mean_iti_s=float(params["meanITI"]) / 1000,
        cs_reward_delay_s=float(params["CS_t_fxd"].strip("[]").split()[1]) / 1000,
        session_description=f["session_description"][()].decode(),
        fiber_location=f["processing/photometry/fibers/location"][0].decode(),
    )
    f.close()
    return out


def main():
    OUT.mkdir(exist_ok=True)
    ds = DandiAPIClient().get_dandiset(DANDISET, "draft")
    assets = sorted((a for a in ds.get_assets() if group_of(a.path) in GROUPS), key=lambda a: a.path)
    print(f"{len(assets)} sessions from {len({a.path.split('/')[0] for a in assets})} mice")

    index = []
    for a in tqdm(assets, desc="Loading sessions"):
        name = Path(a.path).stem  # e.g. sub-60sD-F7_ses-day01
        subject, session = name.split("_")
        dest = OUT / f"{name}.npz"
        if not dest.exists():
            np.savez_compressed(dest, **read_session(a.get_content_url(follow_redirects=1, strip_query=True)))
        sess = np.load(dest)
        index.append(
            dict(
                path=a.path,
                file=dest.name,
                subject=subject.removeprefix("sub-"),
                session=session.removeprefix("ses-"),
                group=group_of(a.path),
                mean_iti_s=float(sess["mean_iti_s"]),
                cs_reward_delay_s=float(sess["cs_reward_delay_s"]),
                n_cs=int(len(sess["cs"])),
                n_reward=int(len(sess["reward"])),
                n_lick=int(len(sess["lick"])),
                duration_s=float(sess["dff_t"][-1]),
                session_description=str(sess["session_description"]),
                fiber_location=str(sess["fiber_location"]),
            )
        )
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
