from __future__ import annotations

from functools import cache


@cache
def sdr_present() -> bool:
    try:
        import SoapySDR
    except ImportError:
        return False
    try:
        return bool(SoapySDR.Device.enumerate(""))
    except Exception:
        return False
