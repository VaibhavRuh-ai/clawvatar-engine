"""Provider registry — maps provider names to implementations."""

from __future__ import annotations

from clawvatar.providers.base import LipSyncProvider


def get_lipsync(name: str) -> LipSyncProvider:
    if name == "rhubarb":
        from clawvatar.lipsync.rhubarb import RhubarbLipSync
        return RhubarbLipSync()
    elif name == "energy":
        from clawvatar.lipsync.energy import EnergyLipSync
        return EnergyLipSync()
    else:
        raise ValueError(f"Unknown lip-sync provider: {name}. Available: rhubarb, energy")
