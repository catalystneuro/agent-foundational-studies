#!/usr/bin/env python3
"""Convert jupytext Python script to Jupyter notebook."""

import subprocess
import sys

# Try to convert using jupytext
script_file = "theta_phase_entrainment_from_file.py"
notebook_file = "theta_phase_entrainment.ipynb"

print(f"Converting {script_file} to {notebook_file}...")

try:
    # Use jupytext to convert
    result = subprocess.run(
        ["jupytext", "--to", "notebook", script_file, "-o", notebook_file],
        check=True,
        capture_output=True,
        text=True
    )
    print("Conversion successful!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error during conversion: {e}")
    print(e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print("jupytext not found, trying nbconvert approach...")
    # Alternative: use nbconvert with a temp approach
    try:
        result = subprocess.run(
            ["python3", "-m", "nbconvert", "--to", "notebook", "--execute", script_file],
            check=False,
            capture_output=True,
            text=True
        )
        print("Note: Direct conversion may not work with Python scripts. Consider using jupytext.")
    except Exception as e:
        print(f"Alternative conversion also failed: {e}")
        sys.exit(1)

print(f"Output notebook: {notebook_file}")
