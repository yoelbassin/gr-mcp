"""GNU Radio sample engine. The only module that imports gnuradio.

All GNU Radio imports happen inside functions so that `import marconi`
works on machines without GNU Radio.
"""

import logging
import math
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from marconi.backends.base import Backend, BackendError
from marconi.models import DeviceInfo, PipelineSpec, RunResult

logger = logging.getLogger(__name__)

_SINK_TYPES = {"file_sink", "wav_sink"}


def _modules() -> Any:
    try:
        from gnuradio import analog, blocks
        from gnuradio import filter as gr_filter
        from gnuradio import gr
        from gnuradio.filter import firdes
    except ImportError as e:  # pragma: no cover
        raise BackendError(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "(https://wiki.gnuradio.org/index.php/InstallingGR)."
        ) from e
    return gr, blocks, analog, gr_filter, firdes


def _factories(rate: float) -> dict[str, Callable[[dict[str, Any]], Any]]:
    gr, blocks, analog, gr_filter, firdes = _modules()

    def r(p: dict[str, Any]) -> float:
        return float(p.get("sample_rate", rate))

    return {
        "tone_source": lambda p: analog.sig_source_c(
            r(p), analog.GR_COS_WAVE, p["freq"], p.get("amplitude", 1.0), 0
        ),
        "audio_tone_source": lambda p: analog.sig_source_f(
            r(p), analog.GR_COS_WAVE, p["freq"], p.get("amplitude", 0.5), 0
        ),
        "noise_source": lambda p: analog.noise_source_c(
            analog.GR_GAUSSIAN, p["amplitude"], p.get("seed", 0)
        ),
        "file_source": lambda p: blocks.file_source(
            gr.sizeof_gr_complex, p["path"], p.get("repeat", False)
        ),
        "head": lambda p: blocks.head(gr.sizeof_gr_complex, int(p["num_samples"])),
        "add": lambda p: blocks.add_vcc(1),
        "multiply_const": lambda p: blocks.multiply_const_cc(p["value"]),
        "freq_shift": lambda p: blocks.rotator_cc(2.0 * math.pi * p["offset"] / r(p)),
        "freq_xlating_lowpass": lambda p: gr_filter.freq_xlating_fir_filter_ccf(
            int(p["decimation"]),
            firdes.low_pass(1.0, r(p), p["cutoff"], p["transition"]),
            p["center_offset"],
            r(p),
        ),
        "quadrature_demod": lambda p: analog.quadrature_demod_cf(p.get("gain", 1.0)),
        "rational_resampler_f": lambda p: gr_filter.rational_resampler_fff(
            int(p["interpolation"]), int(p["decimation"])
        ),
        "rational_resampler_c": lambda p: gr_filter.rational_resampler_ccc(
            int(p["interpolation"]), int(p["decimation"])
        ),
        "fm_deemphasis": lambda p: analog.fm_deemph(fs=r(p), tau=p.get("tau", 75e-6)),
        "nbfm_rx": lambda p: analog.nbfm_rx(
            audio_rate=int(p["audio_rate"]),
            quad_rate=int(p["quad_rate"]),
            tau=p.get("tau", 75e-6),
            max_dev=p.get("max_dev", 5e3),
        ),
        "nbfm_tx": lambda p: analog.nbfm_tx(
            audio_rate=int(p["audio_rate"]),
            quad_rate=int(p["quad_rate"]),
            tau=p.get("tau", 75e-6),
            max_dev=p.get("max_dev", 5e3),
        ),
        "file_sink": lambda p: blocks.file_sink(gr.sizeof_gr_complex, p["path"], False),
        "wav_sink": lambda p: blocks.wavfile_sink(
            p["path"],
            1,
            int(p["sample_rate"]),
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False,
        ),
    }


def build_top_block(spec: PipelineSpec) -> tuple[Any, list[Path]]:
    """Instantiate a validated PipelineSpec as a gr.top_block.

    Returns (top_block, artifact_paths). Raises BackendError with the
    offending block id on construction failure.
    """
    gr, *_ = _modules()
    factories = _factories(spec.sample_rate)

    tb = gr.top_block(spec.name)
    instances: dict[str, Any] = {}
    artifacts: list[Path] = []

    for b in spec.blocks:
        factory = factories.get(b.type)
        if factory is None:
            raise BackendError(
                f"block '{b.id}': type '{b.type}' has no GNU Radio factory"
            )
        try:
            instances[b.id] = factory(dict(b.params))
        except Exception as e:
            raise BackendError(
                f"block '{b.id}' ({b.type}) failed to construct: {e}"
            ) from e
        if b.type in _SINK_TYPES:
            artifacts.append(Path(str(b.params["path"])))

    for c in spec.connections:
        try:
            tb.connect(
                (instances[c.src_block], c.src_port),
                (instances[c.dst_block], c.dst_port),
            )
        except Exception as e:
            raise BackendError(
                f"connecting {c.src_block}:{c.src_port} -> "
                f"{c.dst_block}:{c.dst_port} failed: {e}"
            ) from e

    return tb, artifacts


def _run_with_timeout(tb: Any, timeout: float, name: str) -> tuple[bool, str | None]:
    """Run `tb` on a daemon worker, bounded by `timeout`. Returns
    (timed_out, failure_traceback) — failure_traceback is None on success."""
    failure: list[str] = []

    def _run() -> None:
        try:
            tb.run()
        except Exception:
            failure.append(traceback.format_exc())

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)

    timed_out = worker.is_alive()
    if timed_out:
        tb.stop()
        tb.wait()
        worker.join(5.0)
        if worker.is_alive():
            logger.warning(
                "GNU Radio worker did not exit within the grace period after "
                "stop(); flowgraph '%s' may be wedged.",
                name,
            )

    return timed_out, failure[0] if failure else None


class GnuRadioBackend(Backend):
    """GNU Radio sample engine."""

    name = "gnuradio"

    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        start = time.monotonic()
        try:
            tb, artifacts = build_top_block(spec)
        except BackendError as e:
            return RunResult(
                status="error", elapsed_seconds=time.monotonic() - start, error=str(e)
            )

        timed_out, failure = _run_with_timeout(tb, timeout, spec.name)
        elapsed = time.monotonic() - start

        if failure is not None:
            return RunResult(
                status="error",
                elapsed_seconds=elapsed,
                artifacts=artifacts,
                error=f"flowgraph raised during run:\n{failure}",
            )
        return RunResult(
            status="timeout" if timed_out else "ok",
            elapsed_seconds=elapsed,
            artifacts=artifacts,
            error="run exceeded timeout and was stopped" if timed_out else None,
        )

    def enumerate_devices(self) -> list[DeviceInfo]:
        return []
