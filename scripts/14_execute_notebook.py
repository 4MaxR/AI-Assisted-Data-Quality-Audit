"""Execute notebooks/churn_analysis.ipynb in place so the submitted notebook
ships with real outputs rather than empty cells.

Uses nbclient directly (no jupyter/nbconvert dependency) with the project venv
kernel, a generous cell timeout, and a hard failure on any cell error so a broken
notebook can never be mistaken for a finished one.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path("notebooks/churn_analysis.ipynb")

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    timeout=1200,
    kernel_name="python3",
    allow_errors=False,          # any exception must fail the build
    resources={"metadata": {"path": str(NB.parent)}},
)

print(f"executing {NB} ...")
client.execute()

nbformat.write(nb, NB)

code_cells = [c for c in nb.cells if c.cell_type == "code"]
with_output = [c for c in code_cells if c.get("outputs")]
print(f"  {len(code_cells)} code cells, {len(with_output)} produced output")
blank = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
if blank:
    print(f"  WARNING: code cells with no output at positions {blank}")
    sys.exit(1)
print("  all code cells executed successfully")
