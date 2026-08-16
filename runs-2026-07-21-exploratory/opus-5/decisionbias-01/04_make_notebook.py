"""Convert the jupytext script to an executed .ipynb."""
import subprocess
import sys

SRC = "decision_bias_prestimulus.py"
DST = "decision_bias_prestimulus.ipynb"

subprocess.run([sys.executable, "-m", "jupytext", "--to", "notebook", "-o", DST, SRC],
               check=True)
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", "--ExecutePreprocessor.timeout=7200", DST], check=True)
print("wrote", DST)
