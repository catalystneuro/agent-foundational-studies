"""Convert the consolidated jupytext script to an executed .ipynb."""
import subprocess
import sys

SRC = "grid_cells_mec_dandi000582.py"
OUT = "grid_cells_mec_dandi000582.ipynb"

subprocess.run([sys.executable, "-m", "jupytext", "--to", "notebook",
                "--output", OUT, SRC], check=True)
print(f"wrote {OUT}")
