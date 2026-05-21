"""Execute a notebook in-place using nbclient.

Equivalent to `jupyter nbconvert --to notebook --execute --inplace <path>` but
without requiring the jupyter CLI to be installed.

Usage:
    python scripts/_execute_notebook.py presentation/backprop_env.ipynb [--kernel amm-gym-venv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path)
    p.add_argument("--kernel", default="amm-gym-venv")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    nb_path = args.notebook.resolve()
    nb = nbformat.read(nb_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()

    # Restore the original kernelspec (we override with `amm-gym-venv` for
    # execution; on disk we keep `python3` to match `realistic_simulator.ipynb`).
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }

    nbformat.write(nb, nb_path)
    print(f"Executed {nb_path} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
