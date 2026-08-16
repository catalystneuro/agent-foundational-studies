#!/usr/bin/env python3
"""Convert jupytext .py file to .ipynb notebook"""

import subprocess
import sys

input_file = "head_direction_analysis.py"
output_file = "head_direction_analysis.ipynb"

print(f"Converting {input_file} to {output_file}...")

try:
    subprocess.run(['jupytext', '--to', 'notebook', input_file, '-o', output_file], check=True)
    print(f"✓ Conversion complete: {output_file}")
except FileNotFoundError:
    print("jupytext not found, installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'jupytext'], check=True)
    subprocess.run(['jupytext', '--to', 'notebook', input_file, '-o', output_file], check=True)
    print(f"✓ Conversion complete: {output_file}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
