"""Import shim: lets 04_visualize.py and 05_multisession.py reuse 03_precession.py,
whose name starts with a digit and so cannot be imported directly."""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_precession_03", pathlib.Path(__file__).with_name("03_precession.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

analyse_session = _mod.analyse_session
SESSION = _mod.SESSION
