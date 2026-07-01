from __future__ import annotations

from multiprocessing.connection import Connection

from marconi.phy.backends.base import BackendError, RunResult
from marconi.phy.ir import GrPipeline

_SINK_KINDS = {
    "iq_file_sink",
    "bits_file_sink",
    "soft_bits_file_sink",
    "symbols_file_sink",
}


def sink_paths(pipeline: GrPipeline) -> list[str]:
    return [
        str(b.params["path"])
        for b in pipeline.blocks
        if b.kind in _SINK_KINDS and "path" in b.params
    ]


def _run_flowgraph(pipeline: GrPipeline) -> RunResult:
    from marconi.phy.backends.gnuradio.build import build_top_block  # lazy

    try:
        tb = build_top_block(pipeline)
    except BackendError as e:
        return RunResult(status="error", error=str(e))
    try:
        tb.run()
    except Exception as e:  # noqa: BLE001
        return RunResult(status="error", error=f"flowgraph raised: {e}")
    return RunResult(status="ok", artifacts=sink_paths(pipeline))


def run_pipeline_worker(payload_json: str, conn: Connection) -> None:
    result = _run_flowgraph(GrPipeline.model_validate_json(payload_json))
    conn.send(result.model_dump_json())
    conn.close()
