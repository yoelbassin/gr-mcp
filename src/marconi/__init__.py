"""Marconi: LLM-driven RF control and analysis."""

from marconi.devices import (
    SimulatedDevice,
    add_simulated_device,
    clear_devices,
    get_device,
    list_devices,
)
from marconi.models import (
    BlockSpec,
    Burst,
    CaptureRef,
    ConnectionSpec,
    DetectedSignal,
    DeviceInfo,
    PipelineSpec,
    PSDResult,
    RenderResult,
    RunResult,
    SceneElement,
    SceneSpec,
    SignalMeasurement,
    SignalPeak,
    ValidationIssue,
)
from marconi.ops.analyze import detect_bursts, find_signals, measure, psd
from marconi.ops.capture import capture, load_capture
from marconi.ops.export_grc import export_grc, export_grc_to_workspace
from marconi.ops.pipeline import run_pipeline, save_pipeline_to_workspace
from marconi.ops.render import constellation, psd_plot, spectrogram
from marconi.ops.simulate import render_scene, scene_to_pipeline
from marconi.ops.transmit import (
    TransmitForbiddenError,
    TransmitNotConfirmedError,
    transmit_capture,
)
from marconi.sigmf import read_capture, write_capture
from marconi.specs import load_pipeline, load_scene, save_pipeline, save_scene
from marconi.vocabulary import VOCABULARY, PipelineValidationError, validate_pipeline
from marconi.workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "BlockSpec",
    "Burst",
    "CaptureRef",
    "ConnectionSpec",
    "DetectedSignal",
    "DeviceInfo",
    "PSDResult",
    "PipelineSpec",
    "PipelineValidationError",
    "RenderResult",
    "RunResult",
    "SceneElement",
    "SceneSpec",
    "SignalMeasurement",
    "SignalPeak",
    "SimulatedDevice",
    "TransmitForbiddenError",
    "TransmitNotConfirmedError",
    "VOCABULARY",
    "ValidationIssue",
    "Workspace",
    "add_simulated_device",
    "capture",
    "clear_devices",
    "constellation",
    "detect_bursts",
    "export_grc",
    "export_grc_to_workspace",
    "find_signals",
    "get_device",
    "list_devices",
    "load_capture",
    "load_pipeline",
    "load_scene",
    "measure",
    "psd",
    "psd_plot",
    "read_capture",
    "render_scene",
    "run_pipeline",
    "save_pipeline",
    "save_pipeline_to_workspace",
    "save_scene",
    "scene_to_pipeline",
    "spectrogram",
    "transmit_capture",
    "validate_pipeline",
    "write_capture",
]
