# %% [markdown]
# # DANDI 000044 — Hippocampal Place Cells in the Achilles Session
#
# This notebook loads a single linear-track recording session from the
# Buzsaki lab dandiset ("Diversity in neural firing dynamics supports both rigid
# and learned hippocampal sequences", DANDI:000044) and demonstrates the
# canonical signature of hippocampal place cells: spatially selective firing
# along the 1.6 m linear maze.
#
# Data access uses remfile streaming with a local disk cache, so the 8.7 GB NWB
# file is never downloaded in full.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import scipy.signal
from tqdm import tqdm

print("pynapple", nap.__version__)

# path to the presigned S3 url (written by 01_resolve_url.py)
s3_url = open("s3_url.txt").read().strip()

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

io.close()