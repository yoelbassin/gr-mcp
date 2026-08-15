"""capture's failure branches through the REAL backend, with no SDR attached.

The rest of the tree drives these branches with a stub driver, and a stub that
agrees with a wrong idea of what the runner returns proves nothing about the
contract the tool documents. A source for a device that does not exist fails
inside the worker on any machine, radio plugged in or not, so the error
arrives from the real runner: worker exit, RunStatus, empty sink, refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers._capture import stub_probe

from marconi.capture import CaptureError
from marconi.capture import record as record_mod
from marconi.capture.record import capture_iq
from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.errors import classify_error

_ABSENT_DEVICE = "driver=marconi_no_such_driver"


def test_a_run_that_never_opened_the_radio_refuses_rather_than_reporting_a_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_worker_warm()
    monkeypatch.setattr(record_mod, "_probe_device", stub_probe)
    out = tmp_path / "iq.cf32"
    with pytest.raises(CaptureError) as excinfo:
        capture_iq(
            out,
            center_hz=100e6,
            sample_rate=2.048e6,
            duration_s=0.05,
            device=_ABSENT_DEVICE,
        )
    code, message = classify_error(excinfo.value)
    assert code == "failed_precondition"
    assert "no samples" in message
    assert not out.exists() or out.stat().st_size == 0
