"""Run STRF analysis over all selected sessions."""
import sys, time
from analyze_session import analyze_session
from session_strf import ASSETS

for label, aid in ASSETS.items():
    t0 = time.time()
    analyze_session(aid, label)
    print(f'=== {label} total {time.time()-t0:.0f}s ===', flush=True)
print('ALL DONE')
