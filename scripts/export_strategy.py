"""Render a submittable Strategy.sol from frozen reactive-controller params."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_policies import ReactiveControllerParams
TEMPLATE_PATH = ROOT / "contracts" / "generated" / "ReactiveStrategyTemplate.sol"
DEFAULT_OUTPUT = ROOT / "contracts" / "generated" / "Strategy.sol"


def render_strategy_source(params: ReactiveControllerParams) -> str:
    source = TEMPLATE_PATH.read_text()
    replacements = {
        "__BASE_FEE__": str(params.to_wad_dict()["base_fee"]),
        "__BASE_SPREAD__": str(params.to_wad_dict()["base_spread"]),
        "__FLOW_DECAY__": str(params.to_wad_dict()["flow_decay"]),
        "__SIZE_DECAY__": str(params.to_wad_dict()["size_decay"]),
        "__GAP_DECAY__": str(params.to_wad_dict()["gap_decay"]),
        "__TOXICITY_DECAY__": str(params.to_wad_dict()["toxicity_decay"]),
        "__SIZE_WEIGHT__": str(params.to_wad_dict()["size_weight"]),
        "__GAP_WEIGHT__": str(params.to_wad_dict()["gap_weight"]),
        "__GAP_TARGET__": str(params.to_wad_dict()["gap_target"]),
        "__FLOW_TO_MID__": str(params.to_wad_dict()["flow_to_mid"]),
        "__FLOW_TO_SPREAD__": str(params.to_wad_dict()["flow_to_spread"]),
        "__FLOW_TO_SKEW__": str(params.to_wad_dict()["flow_to_skew"]),
        "__TOXICITY_TO_MID__": str(params.to_wad_dict()["toxicity_to_mid"]),
        "__TOXICITY_TO_SIDE__": str(params.to_wad_dict()["toxicity_to_side"]),
        "__BUY_TOXICITY_WEIGHT__": str(params.to_wad_dict()["buy_toxicity_weight"]),
        "__SELL_TOXICITY_WEIGHT__": str(params.to_wad_dict()["sell_toxicity_weight"]),
    }
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    return source


def load_params(path: Path) -> ReactiveControllerParams:
    data = json.loads(path.read_text())
    if "params" in data:
        data = data["params"]
    return ReactiveControllerParams(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a submittable reactive Strategy.sol")
    parser.add_argument("params_json", type=Path, help="Path to frozen parameter JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output Solidity file")
    args = parser.parse_args()

    params = load_params(args.params_json).normalized()
    source = render_strategy_source(params)
    args.output.write_text(source)
    print(args.output)


if __name__ == "__main__":
    main()
