"""Load one session of DANDI:000582 (Sargolini et al. 2006) and inspect its contents."""
import sys
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

DANDISET, VERSION = "000582", "0.251111.2151"
CACHE = "/tmp/remfile_cache"


def get_url(path):
    with DandiAPIClient() as c:
        return c.get_dandiset(DANDISET, VERSION).get_asset_by_path(path).get_content_url(follow_redirects=1, strip_query=True)


def load(path):
    rf = remfile.File(get_url(path), disk_cache=remfile.DiskCache(CACHE))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile), io


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sub-11207/sub-11207_ses-08060501_behavior+ecephys.nwb"
    nwbfile, nwb, io = load(path)
    print(nwbfile)
    print(nwbfile.session_description, nwbfile.experiment_description)
    print(nwb)
    print(nwbfile.units.to_dataframe().drop(columns="spike_times").head(20))
    for k in nwbfile.processing:
        print(k, list(nwbfile.processing[k].data_interfaces))
        for n, di in nwbfile.processing[k].data_interfaces.items():
            print(n, di)
