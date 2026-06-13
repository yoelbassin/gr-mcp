"""The MCP tool functions: thin, synchronous marshalling over the marconi ops.

Each function takes JSON-friendly args, calls into the marconi library, and
returns plain dicts/lists. Captures are referenced by their .sigmf-data path
string (CaptureRef.path); _ref() reconstructs the CaptureRef from the sidecar.
Every function is wrapped by tool_error_boundary and registered in TOOLS."""

from __future__ import annotations

from collections.abc import Callable

import marconi
from marconi import sigmf
from marconi.mcp.errors import tool_error_boundary
from marconi.mcp.state import get_state
from marconi.models import CaptureRef, PipelineSpec, SceneElement, SceneSpec
from marconi.specs import save_scene
from marconi.vocabulary import VOCABULARY


def _ref(capture_path: str) -> CaptureRef:
    """Reconstruct a CaptureRef from a capture's .sigmf-data path."""
    return sigmf.read_meta(capture_path)


@tool_error_boundary
def list_blocks() -> dict:
    """The curated block vocabulary for composing pipelines. For each block
    type: input/output dtypes ('c'=complex, 'f'=float) and its parameters
    (name, type, required, default). Compose pipelines ONLY from these types."""
    out: dict = {}
    for name, d in VOCABULARY.items():
        out[name] = {
            "inputs": list(d.inputs),
            "outputs": list(d.outputs),
            "params": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "required": p.required,
                    "default": p.default,
                }
                for p in d.params
            ],
        }
    return out


@tool_error_boundary
def list_devices() -> list[dict]:
    """List available devices: simulated devices plus any backend hardware
    (none in v1.0). Each entry has id, kind, can_tx, description."""
    return [d.model_dump() for d in marconi.list_devices()]


@tool_error_boundary
def simulate_scene(
    device_id: str, elements: list[dict], scene_name: str | None = None
) -> dict:
    """Register a simulated device whose 'on-air' contents are `elements`.

    Each element: {kind: tone|noise|fm_tone|iq_file, freq: Hz (absolute,
    ignored for noise), amplitude: float, params: {...}}. fm_tone needs
    params.mod_freq and renders only when the capture sample_rate is a multiple
    of 100000. The scene is persisted to scenes/<device_id>.yaml so the device
    survives a restart. Always include a small noise element."""
    state = get_state()
    scene = SceneSpec(
        name=scene_name or device_id,
        elements=[SceneElement(**e) for e in elements],
    )
    dev = marconi.add_simulated_device(device_id, scene)
    save_scene(scene, state.workspace.root / "scenes" / f"{device_id}.yaml")
    return dev.info().model_dump()


@tool_error_boundary
def load_capture(
    path: str, sample_rate: float | None = None, center_freq: float | None = None
) -> dict:
    """Ingest an external IQ file (SigMF / .cf32 / .wav) into the workspace.
    .cf32 requires sample_rate. Returns a capture reference."""
    ref = marconi.load_capture(
        path, get_state().workspace, sample_rate=sample_rate, center_freq=center_freq
    )
    return ref.model_dump(mode="json")


@tool_error_boundary
def capture(
    device_id: str,
    center_freq: float,
    sample_rate: float,
    duration: float,
    name: str | None = None,
) -> dict:
    """Capture IQ from a device (v1.0: simulated) as seen at center_freq /
    sample_rate for `duration` seconds. Returns a capture reference whose
    'path' feeds the analyze/render tools."""
    ref = marconi.capture(
        device_id,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=get_state().workspace,
        name=name,
    )
    return ref.model_dump(mode="json")


@tool_error_boundary
def psd(capture_path: str, nperseg: int = 4096) -> dict:
    """Power-spectral-density summary: the noise floor (dB) and the strongest
    spectral peaks (freq Hz, power dB). The full PSD curve is intentionally
    omitted to keep responses small — render psd_plot and read the image for
    the shape."""
    r = marconi.psd(_ref(capture_path), nperseg=nperseg)
    return {
        "noise_floor_db": r.noise_floor_db,
        "peaks": [p.model_dump() for p in r.peaks],
        "num_bins": len(r.freqs),
    }


@tool_error_boundary
def find_signals(
    capture_path: str,
    threshold_db: float = 6.0,
    min_bandwidth: float = 500.0,
    nperseg: int = 4096,
) -> list[dict]:
    """Detect signals as contiguous PSD regions above the noise floor. Each:
    center_freq, bandwidth (threshold-crossing extent), peak_power_db, snr_db.
    Note: wideband signals (e.g. FM) with a low noise floor can fragment into
    several detections — cross-check with measure() and a spectrogram."""
    return [
        s.model_dump()
        for s in marconi.find_signals(
            _ref(capture_path),
            threshold_db=threshold_db,
            min_bandwidth=min_bandwidth,
            nperseg=nperseg,
        )
    ]


@tool_error_boundary
def measure(
    capture_path: str,
    center_freq: float,
    search_bandwidth: float = 200e3,
    nperseg: int = 4096,
) -> dict:
    """Measure the signal nearest center_freq within search_bandwidth:
    center_freq, occupied_bw_99 (99% power bandwidth), power_db, snr_db. Treat a
    signal as reliably present only above ~8 dB SNR."""
    return marconi.measure(
        _ref(capture_path),
        center_freq=center_freq,
        search_bandwidth=search_bandwidth,
        nperseg=nperseg,
    ).model_dump()


@tool_error_boundary
def detect_bursts(
    capture_path: str, window: float = 1e-3, threshold_db: float = 6.0
) -> list[dict]:
    """Detect on/off bursts from the smoothed power envelope. Each: start_time,
    duration (s), mean_power_db. An always-on signal yields no bursts; valid for
    duty cycles below ~75%."""
    return [
        b.model_dump()
        for b in marconi.detect_bursts(
            _ref(capture_path), window=window, threshold_db=threshold_db
        )
    ]


@tool_error_boundary
def spectrogram(capture_path: str, name: str = "spectrogram", nfft: int = 1024) -> dict:
    """Render a spectrogram PNG into the workspace. Returns {path, kind}; read
    the image with vision to see signals over time and frequency."""
    return marconi.spectrogram(
        _ref(capture_path), get_state().workspace, name=name, nfft=nfft
    ).model_dump(mode="json")


@tool_error_boundary
def psd_plot(capture_path: str, name: str = "psd") -> dict:
    """Render a PSD plot PNG (power vs frequency, noise floor, top peaks)."""
    return marconi.psd_plot(
        _ref(capture_path), get_state().workspace, name=name
    ).model_dump(mode="json")


@tool_error_boundary
def constellation(
    capture_path: str, name: str = "constellation", max_points: int = 5000
) -> dict:
    """Render an I/Q constellation PNG (useful after channelizing/demodulating)."""
    return marconi.constellation(
        _ref(capture_path), get_state().workspace, name=name, max_points=max_points
    ).model_dump(mode="json")


@tool_error_boundary
def validate_pipeline(pipeline: dict) -> list[dict]:
    """Validate a pipeline spec against the vocabulary. Returns a list of issues
    (empty = valid); each issue names the block_id and field. Always validate
    and fix all issues before running."""
    spec = PipelineSpec.model_validate(pipeline)
    return [i.model_dump() for i in marconi.validate_pipeline(spec)]


@tool_error_boundary
def run_pipeline(pipeline: dict, timeout: float = 30.0) -> dict:
    """Validate then run a pipeline (blocks until it finishes or `timeout`
    seconds elapse). Returns {run_id, pipeline, status (ok|timeout|error),
    elapsed_seconds, artifacts, error}. Invalid specs raise a validation error."""
    state = get_state()
    spec = PipelineSpec.model_validate(pipeline)
    result = marconi.run_pipeline(spec, timeout=timeout)
    return state.record_run(state.next_run_id(), spec.name, result)


@tool_error_boundary
def save_pipeline(pipeline: dict) -> dict:
    """Save a pipeline spec as YAML under workspace/pipelines/. Returns {path}."""
    spec = PipelineSpec.model_validate(pipeline)
    path = marconi.save_pipeline_to_workspace(spec, get_state().workspace)
    return {"path": str(path)}


@tool_error_boundary
def export_grc(pipeline: dict, name: str | None = None) -> dict:
    """Export a pipeline as a GNU Radio Companion .grc file under
    workspace/pipelines/ so the user can open and tweak it in GRC. Returns {path}."""
    state = get_state()
    spec = PipelineSpec.model_validate(pipeline)
    out = state.workspace.root / "pipelines" / f"{name or spec.name}.grc"
    out.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(marconi.export_grc(spec, out))}


@tool_error_boundary
def list_runs() -> list[dict]:
    """The history of pipeline runs this session: each {run_id, pipeline,
    status, elapsed_seconds, artifacts, error}."""
    return list(get_state().runs)


@tool_error_boundary
def render_scene(
    elements: list[dict],
    center_freq: float,
    sample_rate: float,
    duration: float,
    scene_name: str = "scene",
    name: str | None = None,
) -> dict:
    """Render an ad-hoc scene (inline `elements`, no device registered) to a
    capture in the workspace. Use simulate_scene + capture when you want a
    persistent device. Returns a capture reference."""
    scene = SceneSpec(name=scene_name, elements=[SceneElement(**e) for e in elements])
    return marconi.render_scene(
        scene,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=get_state().workspace,
        name=name,
    ).model_dump(mode="json")


@tool_error_boundary
def transmit_capture(
    device_id: str,
    capture_path: str,
    freq: float,
    amplitude: float = 1.0,
    confirmed: bool = False,
) -> dict:
    """Replay a capture 'on the air' into a simulated device's scene (it becomes
    an iq_file element, audible to later captures at the SAME sample_rate).
    Gated by CONFIRM_TX: pass confirmed=True only after checking freq and
    device. The change is session-scoped (not re-persisted to the scene file)."""
    el = marconi.transmit_capture(
        device_id,
        _ref(capture_path),
        freq=freq,
        amplitude=amplitude,
        confirmed=confirmed,
    )
    return el.model_dump(mode="json")


# Tools are added to this registry as later tasks implement them.
TOOLS: dict[str, Callable] = {
    "list_blocks": list_blocks,
    "list_devices": list_devices,
    "simulate_scene": simulate_scene,
    "load_capture": load_capture,
    "capture": capture,
    "psd": psd,
    "find_signals": find_signals,
    "measure": measure,
    "detect_bursts": detect_bursts,
    "spectrogram": spectrogram,
    "psd_plot": psd_plot,
    "constellation": constellation,
    "validate_pipeline": validate_pipeline,
    "run_pipeline": run_pipeline,
    "save_pipeline": save_pipeline,
    "export_grc": export_grc,
    "list_runs": list_runs,
    "render_scene": render_scene,
    "transmit_capture": transmit_capture,
}
