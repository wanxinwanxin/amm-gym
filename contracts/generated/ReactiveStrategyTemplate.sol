// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AMMStrategyBase} from "./AMMStrategyBase.sol";
import {IAMMStrategy, TradeInfo} from "./IAMMStrategy.sol";

// Parameter placeholders are injected by scripts/export_strategy.py.
contract Strategy is AMMStrategyBase {
    uint256 internal constant BASE_FEE = __BASE_FEE__;
    uint256 internal constant BASE_SPREAD = __BASE_SPREAD__;
    uint256 internal constant FLOW_DECAY = __FLOW_DECAY__;
    uint256 internal constant SIZE_DECAY = __SIZE_DECAY__;
    uint256 internal constant GAP_DECAY = __GAP_DECAY__;
    uint256 internal constant TOXICITY_DECAY = __TOXICITY_DECAY__;
    int256 internal constant SIZE_WEIGHT = __SIZE_WEIGHT__;
    int256 internal constant GAP_WEIGHT = __GAP_WEIGHT__;
    uint256 internal constant GAP_TARGET = __GAP_TARGET__;
    int256 internal constant FLOW_TO_MID = __FLOW_TO_MID__;
    uint256 internal constant FLOW_TO_SPREAD = __FLOW_TO_SPREAD__;
    int256 internal constant FLOW_TO_SKEW = __FLOW_TO_SKEW__;
    uint256 internal constant TOXICITY_TO_MID = __TOXICITY_TO_MID__;
    uint256 internal constant TOXICITY_TO_SIDE = __TOXICITY_TO_SIDE__;
    uint256 internal constant BUY_TOXICITY_WEIGHT = __BUY_TOXICITY_WEIGHT__;
    uint256 internal constant SELL_TOXICITY_WEIGHT = __SELL_TOXICITY_WEIGHT__;

    function afterInitialize(uint256, uint256) external override returns (uint256 bidFee, uint256 askFee) {
        slots[0] = WAD; // gap ema
        slots[1] = 0;   // buy flow
        slots[2] = 0;   // sell flow
        slots[3] = 0;   // size ema
        slots[4] = 0;   // toxicity bid
        slots[5] = 0;   // toxicity ask
        slots[6] = 0;   // last timestamp
        return _fees();
    }

    function afterSwap(TradeInfo calldata trade) external override returns (uint256 bidFee, uint256 askFee) {
        uint256 dt = trade.timestamp > slots[6] ? trade.timestamp - slots[6] : 1;
        uint256 reserveY = trade.reserveY == 0 ? 1 : trade.reserveY;
        uint256 sizeRatio = wdiv(trade.amountY, reserveY);

        slots[1] = wmul(slots[1], FLOW_DECAY);
        slots[2] = wmul(slots[2], FLOW_DECAY);
        if (trade.isBuy) {
            slots[1] += wmul(WAD - FLOW_DECAY, sizeRatio);
        } else {
            slots[2] += wmul(WAD - FLOW_DECAY, sizeRatio);
        }
        slots[3] = _decayMix(slots[3], SIZE_DECAY, sizeRatio);
        slots[0] = _decayMix(slots[0], GAP_DECAY, dt * WAD);
        slots[4] = wmul(slots[4], TOXICITY_DECAY);
        slots[5] = wmul(slots[5], TOXICITY_DECAY);

        if (trade.isBuy) {
            slots[4] += wmul(BUY_TOXICITY_WEIGHT, sizeRatio);
        } else {
            slots[5] += wmul(SELL_TOXICITY_WEIGHT, sizeRatio);
        }

        slots[6] = trade.timestamp;
        return _fees();
    }

    function getName() external pure override returns (string memory) {
        return "Reactive Controller";
    }

    function _fees() internal view returns (uint256 bidFee, uint256 askFee) {
        uint256 gapPressure = slots[0] < GAP_TARGET ? GAP_TARGET - slots[0] : 0;
        uint256 flowTotal = slots[1] + slots[2];
        uint256 toxicity = slots[4] + slots[5];
        int256 flowSkew = int256(slots[1]) - int256(slots[2]);

        int256 mid = int256(BASE_FEE)
            + _wmulSigned(SIZE_WEIGHT, int256(slots[3]))
            + _wmulSigned(GAP_WEIGHT, int256(gapPressure))
            + _wmulSigned(FLOW_TO_MID, int256(flowTotal))
            + int256(wmul(TOXICITY_TO_MID, toxicity));
        uint256 spread = BASE_SPREAD + wmul(FLOW_TO_SPREAD, flowTotal) + wmul(TOXICITY_TO_SIDE, toxicity);
        int256 skew = _wmulSigned(FLOW_TO_SKEW, flowSkew);

        bidFee = _clampSigned(mid + int256(spread / 2) + int256(wmul(TOXICITY_TO_SIDE, slots[4])) - skew);
        askFee = _clampSigned(mid + int256(spread / 2) + int256(wmul(TOXICITY_TO_SIDE, slots[5])) + skew);
    }

    function _decayMix(uint256 oldValue, uint256 decay, uint256 observation) internal pure returns (uint256) {
        return wmul(decay, oldValue) + wmul(WAD - decay, observation);
    }

    function _wmulSigned(int256 x, int256 y) internal pure returns (int256) {
        return (x * y + int256(WAD / 2)) / int256(WAD);
    }

    function _clampSigned(int256 value) internal pure returns (uint256) {
        if (value < 0) return 0;
        if (value > int256(MAX_FEE)) return MAX_FEE;
        return uint256(value);
    }
}
