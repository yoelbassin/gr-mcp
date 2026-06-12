"""Marconi: LLM-driven RF control and analysis."""

from marconi.models import (
    Burst,
    CaptureRef,
    DetectedSignal,
    PSDResult,
    RenderResult,
    SignalMeasurement,
    SignalPeak,
)
from marconi.ops.analyze import detect_bursts, find_signals, measure, psd
from marconi.ops.capture import load_capture
from marconi.ops.render import constellation, psd_plot, spectrogram
from marconi.sigmf import read_capture, write_capture
from marconi.workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "Burst",
    "CaptureRef",
    "DetectedSignal",
    "PSDResult",
    "RenderResult",
    "SignalMeasurement",
    "SignalPeak",
    "Workspace",
    "constellation",
    "detect_bursts",
    "find_signals",
    "load_capture",
    "measure",
    "psd",
    "psd_plot",
    "read_capture",
    "spectrogram",
    "write_capture",
]
