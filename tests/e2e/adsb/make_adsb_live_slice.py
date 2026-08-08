"""Carve the live ADS-B off-air slice tests/e2e/adsb/test_adsb_offair.py
decodes: 5 s of complex64 out of a one-time RTL-SDR capture run. The source
run directory is a runtime capture artifact that may be cleaned up later, so
this script documents the carve for as long as it survives; the sliced
adsb_live_2msps_5s.cf32 (gitignored, under artifacts/assets/ADS-B/) is what
the gate actually depends on - run this once and the asset persists.

Provenance: 2026-08-08 live RTL-SDR capture, 1090 MHz, 2 Msps, Tel Aviv coast;
aircraft 8965C2/738284 observed; oracle 12 strict / 7 unique CRC frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "marconi-runs" / "capture-08ac18f4" / "iq.cf32"
_OUT = _ROOT / "artifacts" / "assets" / "ADS-B" / "adsb_live_2msps_5s.cf32"
_START, _STOP = 8_000_000, 18_000_000  # samples: 5 s @ 2 Msps


def main() -> None:
    if not _SRC.exists():
        raise FileNotFoundError(
            f"{_SRC} absent - this one-time capture run dir may already be "
            f"cleaned up; {_OUT} is unaffected if it's already been carved"
        )
    sig = np.fromfile(_SRC, dtype=np.complex64)
    assert sig.size >= _STOP, f"capture too short: {sig.size} samples"

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    sig[_START:_STOP].tofile(_OUT)
    print(
        f"wrote {_OUT}: {_OUT.stat().st_size} bytes "
        f"({_STOP - _START} samples at 2 Msps)"
    )


if __name__ == "__main__":
    main()
