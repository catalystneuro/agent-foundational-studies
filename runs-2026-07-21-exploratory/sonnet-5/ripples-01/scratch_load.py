import time
import numpy as np
import lindi
from pynwb import NWBHDF5IO

URL = 'https://api.dandiarchive.org/api/assets/82714afb-724f-4e2b-b102-c9c47b5cba73/download/'

def load_nwb():
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_hdf5_file(URL, local_cache=local_cache)
    io = NWBHDF5IO(file=f, mode='r')
    nwbfile = io.read()
    return nwbfile, io

if __name__ == '__main__':
    t0 = time.time()
    nwbfile, io = load_nwb()
    print('loaded in', time.time() - t0)

    epochs = nwbfile.epochs.to_dataframe()
    print(epochs)

    lfp = nwbfile.processing['ecephys']['LFP']['LFP']
    fs = lfp.rate
    print('fs', fs)

    # grab 5 minutes near start of PRE epoch (avoid very edge) for all 128 channels
    t_start = 1000.0
    t_dur = 300.0
    i0 = int(t_start * fs)
    i1 = int((t_start + t_dur) * fs)
    t0 = time.time()
    chunk = lfp.data[i0:i1, :]
    print('fetched', chunk.shape, 'in', time.time() - t0)
    np.save('scratch_chunk_prewindow.npy', chunk)
    print('saved')
