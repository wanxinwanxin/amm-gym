"""Surgical edit of presentation/calibration_three_targets.ipynb.

Removes the §3 (T3 correction) story, tightens §2 with a brief
justification of why T3 is USD-weighted, renumbers §4/§5, and drops the
trailing empty cell. Doesn't run cells — pipe through nbclient afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
NB_PATH = WORKTREE / "presentation" / "calibration_three_targets.ipynb"


INTRO_NEW = """# Three-Target Calibration of the Realistic AMM Simulator

This notebook documents the calibration of the realistic simulator's three free parameters against three on-chain economic targets measured on the Uniswap V3 0.05% WETH/USDC pool (the "5bp pool"). The calibration was run offline; here we present the targets, the converged parameter set, and a held-out distribution check on the calibrated simulator.

**Why calibrate.** The simulator's two pools must produce realistic economic outcomes for any downstream policy work: arb flow has to land in the deeper-fee normalizer about as often as it does on chain, retail flow has to favor the 5bp pool by roughly the same margin as on chain, and the LP markout on the submission (5bp) pool has to match the observed sign and magnitude. Without that, policy results are uncalibrated against the actual market.

**Free parameters.**
- `submission_depth_y` — virtual reserve of the 5bp pool, in USDC.
- `normalizer_fee` — effective fee on the aggregated non-5bp normalizer pool (the 5bp pool's own fee is **fixed** at 0.0005 = 5 bps).
- `normalizer_depth_y` — virtual reserve of the normalizer pool, in USDC.

**Approach.** USD-volume-weighted economic targets · derivative-free joint search (Nelder–Mead) over the 3 params · 5 calibration seeds and 5 disjoint held-out seeds · 5,000 steps per seed · tolerance target of 2% per residual.
"""


SECTION_2_NEW = """---
## Section 2: The three targets

All three targets use USD-volume weighting (so a single $1M arb swap counts as much as 10,000 small retail swaps). Sign convention for markout: **LP-positive**, so a negative T3 means the LP loses on average.

| ID | Name | Empirical value | Source | Sim metric |
|----|------|-----------------|--------|------------|
| T1 | `arb_5bp_share` | 0.33733 | `calibration_artifacts/pool_flow_splits.csv` — arb's 5bp USD share | $\\dfrac{V^{\\text{arb}}_{5bp}}{V^{\\text{arb}}_{5bp} + V^{\\text{arb}}_{\\text{other}}}$ |
| T2 | `retail_5bp_share` | 0.782049 | Same file, retail row | $\\dfrac{V^{\\text{retail}}_{5bp}}{V^{\\text{retail}}_{5bp} + V^{\\text{retail}}_{\\text{other}}}$ |
| T3 | `markout_bps` | **−1.05 bps** | `reports/markout_windows.csv`, 7d USD-weighted next-block markout | $10^{4}\\dfrac{\\text{edge}_{5bp}}{V^{\\text{arb+retail}}_{5bp}}$ across 5 seeds |

**Why USD-weighted everywhere.** T1 and T2 are USD volume shares by construction, so the only economically consistent way to define T3 is also USD-volume-weighted: each dollar of order flow gets one vote, and a single whale trade counts as much as the thousands of small retail trades it represents in volume. The 7d window (2026-05-14 → 2026-05-20) is picked because the calibration episodes are short (~5k steps × ~12s ≈ 17 hours of simulated time per seed), and the longer windows (30d / 90d / 180d / 360d / 730d) all give USD-weighted markouts in the **−0.8 to −1.4 bps** band — so −1.05 bps is also the long-horizon answer, not a window-specific accident. Derivation in `reports/5bp_markout_investigation.md`.
"""


SECTION_4_OLD_TO_NEW = (
    "---\n## Section 4: Calibration results",
    "---\n## Section 3: Calibration results",
)


SECTION_5_OLD_TO_NEW = (
    "---\n## Section 5: Validation",
    "---\n## Section 4: Validation",
)


CONCLUSION_NEW = """**Conclusion.** The joint Nelder–Mead basin lands all three calibration residuals at ≤9.3% and all three held-out residuals at ≤11.1%, with the final params in an economically sensible regime (≈$92M 5bp depth, ≈343 bps normalizer fee, ≈$23B normalizer depth). The remaining gap to the 2% tolerance comes from two structural limits identified in §3: T1's per-seed bimodality (variance-limited; fix is more seeds + longer episodes) and T2's routing ceiling (would require a 4th free parameter — a tunable retail preference for the deeper-fee pool — that's outside the 3-parameter scope of this run). The per-trade markout distribution overlay in §4 shows the calibrated simulator captures the empirical shape well in the bulk, with the negative-skew tail (LP losses on whale-driven adverse selection) preserved."""


CELL_4_NEW_SOURCE = (
    "print(f\"T3 target = {TARGETS['T3_markout_bps']:+.2f} bps  "
    "(7d USD-volume-weighted next-block markout, 5bp WETH/USDC)\")\n"
)


def as_lines(text: str) -> list[str]:
    """Return jupyter-style source: lines with trailing newlines except last."""
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1] != "":
        out.append(parts[-1])
    return out


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    new_cells = []
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))

        # Replace the intro markdown to drop the "+3.6 bps mistake" framing.
        if cell["cell_type"] == "markdown" and src.startswith("# Three-Target Calibration"):
            cell = dict(cell)
            cell["source"] = as_lines(INTRO_NEW)
            new_cells.append(cell)
            continue

        # Drop §3 markdown (cell [5]) and its two plot cells (cells [6], [7]).
        # Use first-line predicates so we don't accidentally match the imports
        # cell, which mentions these helpers in a `from ... import (...)` block.
        first_line = src.split("\n", 1)[0].strip()
        if cell["cell_type"] == "markdown" and src.startswith("---\n## Section 3: The T3 correction"):
            continue
        if cell["cell_type"] == "code" and first_line.startswith("fig, ax = plt.subplots") and "plot_size_bucket_markout(load_size_bucket_markout()" in src:
            continue
        if cell["cell_type"] == "code" and first_line.startswith("fig, ax = plt.subplots") and "plot_markout_windows(load_markout_windows()" in src:
            continue
        # Drop trailing empty code cell.
        if cell["cell_type"] == "code" and src.strip() == "":
            continue

        # Replace §2 markdown.
        if cell["cell_type"] == "markdown" and src.startswith("---\n## Section 2: The three targets"):
            cell = dict(cell)
            cell["source"] = as_lines(SECTION_2_NEW)
            new_cells.append(cell)
            continue

        # Replace the print-block that references §3.
        if cell["cell_type"] == "code" and "T3 target =" in src and "3.637" in src:
            cell = dict(cell)
            cell["source"] = as_lines(CELL_4_NEW_SOURCE)
            cell["outputs"] = []
            cell["execution_count"] = None
            new_cells.append(cell)
            continue

        # Renumber §4 -> §3
        if cell["cell_type"] == "markdown" and src.startswith(SECTION_4_OLD_TO_NEW[0]):
            cell = dict(cell)
            cell["source"] = as_lines(src.replace(SECTION_4_OLD_TO_NEW[0], SECTION_4_OLD_TO_NEW[1]))
            new_cells.append(cell)
            continue

        # Renumber §5 -> §4
        if cell["cell_type"] == "markdown" and src.startswith(SECTION_5_OLD_TO_NEW[0]):
            cell = dict(cell)
            cell["source"] = as_lines(src.replace(SECTION_5_OLD_TO_NEW[0], SECTION_5_OLD_TO_NEW[1]))
            new_cells.append(cell)
            continue

        # Rewrite Conclusion.
        if cell["cell_type"] == "markdown" and src.startswith("**Conclusion.**"):
            cell = dict(cell)
            cell["source"] = as_lines(CONCLUSION_NEW)
            new_cells.append(cell)
            continue

        new_cells.append(cell)

    nb["cells"] = new_cells
    # Reset execution counts and outputs so the notebook is in a "fresh" state
    # before re-running. We'll rerun via _execute_notebook.py next.
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Edited {NB_PATH.relative_to(WORKTREE)}: {len(cells)} -> {len(new_cells)} cells.")
    for i, cell in enumerate(new_cells):
        src = "".join(cell.get("source", []))
        first = src.split("\n", 1)[0][:100] if src else "(empty)"
        print(f"  [{i}] {cell['cell_type']}: {first!r}")


if __name__ == "__main__":
    main()
