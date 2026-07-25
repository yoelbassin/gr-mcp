"""Fetch the off-air FM+RDS capture the RDS gate decodes: 4 s of a real
broadcast station at 250 ksps (cf32), recorded for the PySDR textbook's RDS
chapter and hosted in its companion course repo."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/777arc/498x/raw/master/fm_rds_250k_1Msamples.iq"
DEST = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "RDS"
    / "fm_rds_250k_1Msamples.iq"
)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done: {DEST.stat().st_size} bytes")


if __name__ == "__main__":
    main()
