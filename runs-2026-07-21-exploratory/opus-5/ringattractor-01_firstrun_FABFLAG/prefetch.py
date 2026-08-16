"""Warm the remfile disk cache for all analysed sessions in parallel."""
import warnings, time
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
import hd_lib as H
from run_all_sessions import SESSIONS

def go(s):
    t=time.time(); S=H.load_session(s)
    return f"{s}: {len(S['spikes'])} units, {time.time()-t:.0f}s"

with ThreadPoolExecutor(max_workers=8) as ex:
    for r in ex.map(go, SESSIONS):
        print(r, flush=True)
