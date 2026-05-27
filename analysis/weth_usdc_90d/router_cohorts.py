"""Canonical router cohort definitions for WETH/USDC markout / volume-share
analysis on Ethereum mainnet.

This module is the single source of truth — scripts in scripts/calibration/
import RETAIL_ROUTERS / INTENT_PROTOCOLS from here. The SQL files under
analysis/weth_usdc_90d/sql/ also need to match this list; they each inline
the addresses in an UNNEST([...]) block — keep them in sync by hand or by
regenerating from this module.

Audit history:
  2026-05-27  Initial cleanup based on Pinky's audit (see Slack thread
              uniswapteam.slack.com/archives/C0ATSPHFHMJ/p1779887661822949).
              - Removed 2 Wintermute MM contracts (0xbdb3…f47b6, 0x51c7…02a7f)
                originally mis-labeled as "Banana Gun" / "Maestro-style TG bot".
              - Relabeled 0x0000…1ff3 "0x Settler" -> "0x AllowanceHolder".
                AllowanceHolder is 0x's stable approval chokepoint (analogous
                to Uniswap's Permit2 but 0x-specific). The actual Settler
                contracts are versioned/ephemeral; AllowanceHolder is the
                right capture point for modern 0x retail flow.
              - Added 3 high-confidence retail aggregators: Odos Router v2,
                OpenOcean Exchange Proxy, LI.FI Diamond.
              - Moved CoW GPv2Settlement out of retail_routers into a separate
                intent_protocols cohort because it settles both retail
                intents AND solver/institutional flow.
              - Added UniswapX Reactor to intent_protocols.

Pending TODOs (require verification before adding):
  - 0x1111111254760f7ab3d16433cf12f39d3a2bdecb  1inch AggregationRouterV3 legacy
                                                (verify on Etherscan first)
  - 0x74de5d4FCbf63E00296fd95d33236B9794016631  MetaMask Smart Swap
                                                (verify exact address; varies
                                                 across MM versions)
  - Pull current 0x Settler address from
    https://github.com/0xProject/protocol/tree/development/packages/contract-addresses
    and add alongside AllowanceHolder (pin by block range across upgrades).
  - Permit2 (0x000000000022d473030f116ddee9f6b43ac78ba3) is explicitly NOT
    a router — approval layer only; including it would double-count flow.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Retail routers: user-facing swap UIs + aggregators
# -----------------------------------------------------------------------------
RETAIL_ROUTERS: dict[str, str] = {
    # Uniswap routers (6) — interface defaults across versions
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": "Uniswap Universal Router v1",
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "Uniswap Universal Router v1.2",
    "0x66a9893cc07d91d95644aedd05d03f95e1dba8af": "Uniswap Universal Router v2",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap SwapRouter02",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap SwapRouter v3",
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Uniswap V2Router02",
    # 1inch (3 versions; v3 legacy pending verification — see TODO)
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch v4",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch v5",
    "0x111111125421ca6dc452d289314280a0f8842a65": "1inch v6",
    # 0x (2 contracts: ExchangeProxy + AllowanceHolder)
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x ExchangeProxy",
    "0x0000000000001ff3684f28c67538d4d072c22734": "0x AllowanceHolder",
    # Paraswap (2 versions)
    "0xdef171fe48cf0115b1d80b88dc8eab59176fee57": "Paraswap v5",
    "0x6a000f20005980200259b80c5102003040001068": "Paraswap v6",
    # Kyber (2 versions)
    "0x6131b5fae19ea4f9d964eac0408e4408b66337b5": "Kyber MetaAggregationRouter v2",
    "0x617dee16b86534a5d792a4d7a62fb491b544111e": "Kyber AggregationRouter classic",
    # Wallet-native swap routers
    "0x881d40237659c251811cec9c364ef91dc08d300c": "MetaMask Swap router",
    # New high-confidence adds (Pinky audit 2026-05-27; addresses verified via
    # Etherscan after Pinky's quoted Odos address turned out to be wrong/truncated)
    "0xcf5540fffcdc3d510b18bfca6d2b9987b0772559": "Odos Router V2",
    "0x6352a56caadc4f1e25cd6c75970fa768a3304e64": "OpenOcean Exchange Proxy",
    "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae": "LI.FI Diamond",
}

# -----------------------------------------------------------------------------
# Intent protocols: end-user-signed orders settled by competing solvers/fillers.
# Order origin is retail, but the settlement contract mixes in
# solver/institutional flow — analyse separately from direct routers.
# -----------------------------------------------------------------------------
INTENT_PROTOCOLS: dict[str, str] = {
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41": "CoW GPv2Settlement",
    "0x6000da47483062a0d734ba3dc7576ce6a0b645c4": "UniswapX Exclusive Dutch Order Reactor",
}

# -----------------------------------------------------------------------------
# Removed (do not include — known false positives)
# -----------------------------------------------------------------------------
REMOVED: dict[str, str] = {
    "0xbdb3ba9ffe392549e1f8658dd2630c141fdf47b6": "Wintermute MM (was mis-labeled 'Banana Gun')",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "Wintermute MM (was mis-labeled 'Maestro-style TG bot')",
}


def retail_router_addresses() -> list[str]:
    """Lowercase address list of the retail-router cohort."""
    return list(RETAIL_ROUTERS.keys())


def intent_protocol_addresses() -> list[str]:
    return list(INTENT_PROTOCOLS.keys())
