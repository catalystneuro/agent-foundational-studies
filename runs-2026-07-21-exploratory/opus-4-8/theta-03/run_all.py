from hc11_io import SESSIONS
from run_session import analyze_session

for aid in SESSIONS:
    analyze_session(aid)
