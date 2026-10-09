"""
client_fingerprint.py — Lightning implementation inference from public gossip.

Single source of truth for "which client does this node run?" in every stage of this repo. v1 rules come from feature-bit sets observed in the 2026-10-04 graph and
anchored on operators whose implementation is publicly known (ACINQ = Eclair, Blockstream Store /
PeerSwap = CLN, LNBiG / WalletOfSatoshi / Boltz / bfx = LND). Rules must be re-validated when new
releases change advertised features.

Signals, strongest first:
  1. node_announcement feature bits (implementation-specific bits)
       LND    : 2023 (script-enforced-lease) or 31 (amp)       — LND-only features
       Eclair : 37, or 29 together with 61                       — ACINQ set (v2: 29 alone is
                option_dual_fund, which CLN also advertises — v1 mislabeled ~110 CLN nodes)
       CLN    : 11 together with 39/43, *and* 55 (keysend)      — CLN advertises keysend by default (~96%)
                v3: the same set without 55 is Eclair (Eclair has no native keysend); before 2025 Eclair
                lacked 37/61, so v2 labeled ACINQ and other Eclair nodes as CLN in historical snapshots
       LDK    : 39/43/63 without 11, 2023, 31                   — e.g. LDK-based LSPs
  2. modal cltv_expiry_delta of the node's channel_updates (client defaults: LND 80/40, CLN 34,
     Eclair 144, LDK 72) — weak on its own because operators override it
  3. LND defaults: color #3399ff, alias == pubkey[:20]

Returns (client, confidence, reason); confidence in {"operator", "high", "medium", "low"}.

Scope: public (announced) nodes only. Implementations used mostly behind unannounced channels
(LDK in wallets/services, Eclair in Phoenix) are under-represented in any gossip-based share.
"""

from typing import Iterable, Optional, Tuple

LND_ONLY = {2023, 31}

# Operator attribution: nodes whose implementation is known from who runs them, used when the
# feature set alone is ambiguous. Keep a source note for every entry.
KNOWN_OPERATOR_CLIENTS = {
    # Block maintains LDK; its routing nodes advertise an LDK-like set with non-default CLTV 144.
    "027100442c3b79f606f80f322d98d499eefcb060599efc5d4ecb00209c2cb54190": ("LDK", "block-iad-1: Block (LDK maintainer)"),
    "028a33a72ecb09967e67b08a04179fce74730bbc4cb2544c7d6876d20c6450c5eb": ("LDK", "block-pdx-1: Block (LDK maintainer)"),
    # ACINQ develops Eclair; its node advertised a CLN-like set before 2025 (no 37/61)
    "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f": ("Eclair", "ACINQ (Eclair developer)"),
}
CLTV_DEFAULT = {34: "CLN", 72: "LDK", 144: "Eclair", 80: "LND", 40: "LND"}

RULES_VERSION = "v3-2026-10-06"


def classify_node(feature_bits: Optional[Iterable[int]], color: Optional[str] = None,
                  alias: Optional[str] = None, pub_key: Optional[str] = None,
                  cltv_mode: Optional[float] = None) -> Tuple[str, str, str]:
    if pub_key in KNOWN_OPERATOR_CLIENTS:
        client, note = KNOWN_OPERATOR_CLIENTS[pub_key]
        return client, "operator", f"operator-attributed: {note}"

    bits = set(int(b) for b in (feature_bits if feature_bits is not None else []))

    if bits:
        if bits & LND_ONLY:
            return "LND", "high", "lnd-only feature bit (2023 lease / 31 amp)"
        if 37 in bits or {29, 61} <= bits:
            return "Eclair", "high", "eclair feature bits (37, or 29+61)"
        if 11 in bits and bits & {39, 43}:
            if 55 in bits:
                return "CLN", "high", "cln feature set (11 + 39/43 + keysend 55)"
            return "Eclair", "medium", "cln-like set without keysend (55): eclair"
        if bits & {39, 43, 63} and 11 not in bits:
            # Without cltv 72 (LDK default) or gossip-queries (7) this is a guess by elimination
            conf = "high" if cltv_mode == 72 and 7 in bits else "medium"
            return "LDK", conf, "ldk-like feature set (39/43/63 w/o 11)"
        # Older LND releases predate 2023/31 but keep LND's defaults
        if color == "#3399ff" or (alias and pub_key and alias == pub_key[:20]):
            return "LND", "medium", "lnd default color/alias, generic feature set"
        if cltv_mode in CLTV_DEFAULT:
            return CLTV_DEFAULT[int(cltv_mode)], "low", f"generic feature set, cltv default {int(cltv_mode)}"
        return "Unknown", "low", "generic feature set"

    # No node_announcement features (unannounced metadata): policy defaults only
    if color == "#3399ff" or (alias and pub_key and alias == pub_key[:20]):
        return "LND", "low", "no features; lnd default color/alias"
    if cltv_mode in CLTV_DEFAULT:
        return CLTV_DEFAULT[int(cltv_mode)], "low", f"no features; cltv default {int(cltv_mode)}"
    return "Unknown", "low", "no features"
