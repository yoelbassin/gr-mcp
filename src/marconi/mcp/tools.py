from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend
from marconi.engine.compile.compiler import (
    CompiledPipeline,
    CompileError,
    compile_modem,
    compile_pipeline,
)
from marconi.engine.run import PipelineResult
from marconi.engine.run import run_rx as engine_run_rx
from marconi.engine.stages.base import SpecValidationError
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Symbolstream
from marconi.errors import classify_error
from marconi.mcp.boundary import tool_error_boundary
from marconi.mcp.streams import ensure_cf32, parse_bits, render_page
from marconi.mcp.vocab import ENVELOPE, stage_details, stage_index
from marconi.mcp.workspace import new_run_dir

_START_LEVELS = {"iq": Level.IQ, "symbols": Level.SYMBOLS, "bits": Level.BITS}
_DEFAULT_LEVEL = {"c": "iq", "b": "bits", "s": "symbols", "f": "symbols"}
_ITEM = {"c": ItemType.C, "b": ItemType.B, "s": ItemType.S, "f": ItemType.F}
_ITEM_BYTES = {"b": 1, "s": 2, "f": 4}


def _start_descriptor(item_type: str, level: str | None) -> Descriptor:
    if item_type not in _ITEM:
        raise ValueError(f"input_item_type must be one of {sorted(_ITEM)}")
    level_key = level if level is not None else _DEFAULT_LEVEL[item_type]
    if level_key not in _START_LEVELS:
        raise ValueError(f"input_level must be one of {sorted(_START_LEVELS)}")
    carrier = Carrier.SOFT if item_type == "f" else Carrier.HARD
    return Descriptor(_START_LEVELS[level_key], _ITEM[item_type], carrier)


def _trace_rows(modem: Modem, cp: CompiledPipeline) -> list[dict[str, object]]:
    labels = ["<start>"] + [f"{s.conv}[{i}]" for i, s in enumerate(modem.path)]
    return [
        {
            "after": label,
            "level": desc.level.value,
            "item_type": desc.item_type.value,
            "carrier": desc.carrier.value,
            "amplitude": desc.amplitude.value,
            "order": desc.order,
            "frame_len": desc.frame_len,
            "sample_rate": rate,
        }
        for label, desc, rate in zip(labels, cp.boundaries, cp.rates)
    ]


def _input_stream(path: Path, item_type: str) -> Bitstream | Symbolstream:
    if item_type not in _ITEM_BYTES:
        raise ValueError(
            "input_item_type must be one of ['b', 'f', 's'] with input_path"
        )
    items = path.stat().st_size // _ITEM_BYTES[item_type]
    if item_type == "b":
        return Bitstream(path=path, num_bits=items)
    return Symbolstream(
        path=path, num_symbols=items, item_type=cast(Literal["s", "f"], item_type)
    )


def _stream_summary(result: PipelineResult) -> dict[str, object] | None:
    if result.bitstream is not None:
        return {
            "path": str(result.bitstream.path),
            "item_type": "b",
            "items": result.bitstream.num_bits,
        }
    if result.symbolstream is not None:
        return {
            "path": str(result.symbolstream.path),
            "item_type": result.symbolstream.item_type,
            "items": result.symbolstream.num_symbols,
        }
    return None


def describe_stages(
    family: str | None = None, stage: str | None = None
) -> dict[str, object]:
    """Marconi's stage vocabulary, generated live from the engine registry.

    With no arguments: a compact index of every stage (name, family, levels,
    directions) plus the modem-spec envelope and the level/item-type primer.
    Pass stage=<name> or family=<name> for full per-stage detail: input
    contracts (item type, carrier, amplitude, minimum samples-per-symbol) and
    the JSON schema of the stage's spec parameters. Compose a modem spec as
    {"symbol_rate": <float>, "path": [{"conv": <stage name>, ...params}]};
    check it with validate_modem before running."""
    if stage is not None:
        return {"stages": stage_details([stage]), "envelope": ENVELOPE}
    if family is not None:
        names = [str(e["name"]) for e in stage_index() if e["family"] == family]
        if not names:
            fams = sorted({str(e["family"]) for e in stage_index()})
            raise ValueError(f"unknown family {family!r}; known: {fams}")
        return {"stages": stage_details(names), "envelope": ENVELOPE}
    return {"stages": stage_index(), "envelope": ENVELOPE}


def validate_modem(
    spec: dict[str, Any],
    sample_rate: float,
    direction: str = "rx",
    input_item_type: str = "c",
    input_level: str | None = None,
) -> dict[str, object]:
    """Compile a modem spec without running it — the fast iteration loop.

    Returns {"valid": true, "trace": [...]} where each trace row shows the
    stream after each stage (level, item type, carrier hardness, amplitude
    statistic, symbol-alphabet order, frame length, sample rate), or
    {"valid": false, "errors": [...]} with structured compile errors: a
    failing spec is a normal result here, not an exception. direction is
    "rx" (decode) or "tx" (generate); input_item_type/input_level describe
    the entry stream (defaults: complex IQ)."""
    try:
        modem = Modem.from_spec(spec, step_models())
        cp = compile_pipeline(
            modem,
            stage_registry(),
            direction=direction,
            sample_rate=sample_rate,
            start=_start_descriptor(input_item_type, input_level),
            source_io={"path": "unused"},
            sink_io={"path": "unused"},
        )
    except SpecValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {"code": "invalid_argument", "message": i.message, "at": i.block_id}
                for i in exc.issues
            ],
        }
    except (CompileError, ValidationError, ValueError) as exc:
        code, message = classify_error(exc)
        return {"valid": False, "errors": [{"code": code, "message": message}]}
    return {"valid": True, "trace": _trace_rows(modem, cp)}


def run_rx_tool(
    spec: dict[str, Any],
    sample_rate: float,
    capture_path: str | None = None,
    capture_dtype: str = "cf32",
    input_path: str | None = None,
    input_item_type: str | None = None,
    input_level: str | None = None,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Decode: run a modem spec over an IQ capture (or an existing bit/symbol
    stream for coding-only paths) and return the full pipeline result.

    Pass exactly one of capture_path (raw IQ; capture_dtype one of
    cf32/ci16/ci8/cu8, converted once if not cf32) or input_path (+
    input_item_type b/s/f; f is float32 soft values where bit 1 = NEGATIVE).
    The result carries status, a "stream" summary {path, item_type, items}
    to page with read_stream, windows/marks, per-block census, diagnostics,
    and "quality": a conservative verdict (decoded / uncertain / no_signal)
    with the evidence behind it. Treat only verdict "decoded" as trustworthy
    output; "uncertain" means the path produced no checkable evidence."""
    if (capture_path is None) == (input_path is None):
        raise ValueError("pass exactly one of capture_path or input_path")
    modem = Modem.from_spec(spec, step_models())
    run_dir = new_run_dir("rx")
    if capture_path is not None:
        src = ensure_cf32(Path(capture_path), capture_dtype, run_dir)
        result = engine_run_rx(
            modem,
            stage_registry(),
            sample_rate=sample_rate,
            start=_start_descriptor("c", None),
            workdir=run_dir,
            source_io={"path": str(src)},
            timeout=timeout,
        )
    else:
        if input_item_type is None:
            raise ValueError("input_item_type is required with input_path")
        result = engine_run_rx(
            modem,
            stage_registry(),
            sample_rate=sample_rate,
            start=_start_descriptor(input_item_type, input_level),
            workdir=run_dir,
            input_stream=_input_stream(Path(cast(str, input_path)), input_item_type),
            timeout=timeout,
        )
    payload: dict[str, object] = result.model_dump(mode="json")
    payload["stream"] = _stream_summary(result)
    return payload


def run_tx_tool(
    spec: dict[str, Any],
    sample_rate: float,
    bits: str | None = None,
    bits_path: str | None = None,
    out_path: str | None = None,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Generate: render input bits through a modem spec (direction tx) into a
    complex64 IQ file — simulation output only, nothing is transmitted.

    Pass exactly one of bits (a '0'/'1' string) or bits_path (a file of one
    uint8 bit per byte). The path must compile in the tx direction; coding
    stages do not (the engine executes no tx-side coding), so supply bits
    that already carry any framing/encoding you need. Returns the IQ file
    path, sample count, and per-block census."""
    if (bits is None) == (bits_path is None):
        raise ValueError("pass exactly one of bits or bits_path")
    modem = Modem.from_spec(spec, step_models())
    run_dir = new_run_dir("tx")
    if bits is not None:
        bits_file = run_dir / "tx_bits.u8"
        parse_bits(bits).tofile(bits_file)
    else:
        bits_file = Path(cast(str, bits_path))
    out = Path(out_path) if out_path is not None else run_dir / "out.cf32"
    gr = compile_modem(
        modem,
        stage_registry(),
        direction="tx",
        sample_rate=sample_rate,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(bits_file)},
        sink_io={"path": str(out)},
    )
    r = GnuRadioBackend().run_pipeline(gr, timeout=timeout)
    return {
        "status": r.status,
        "error": r.error,
        "iq_path": str(out),
        "num_samples": out.stat().st_size // 8 if out.is_file() else 0,
        "sample_rate": sample_rate,
        "census": [c.model_dump() for c in r.census],
    }


def read_stream(
    path: str, offset: int = 0, count: int = 4096, item_type: str | None = None
) -> dict[str, object]:
    """Page a decoded stream back as data you can parse directly.

    Bits (.u8) return as a '0'/'1' string — the files are unpacked and frames
    are rarely byte-aligned, so slice the string at any bit offset for your
    framing/CRC/field work. Hard symbols (.i16) return as ints, soft values
    (.f32) as floats (bit 1 = NEGATIVE). item_type b/s/f overrides suffix
    inference (required for suffix-less paths). Pages are capped at 65536
    items; use offset to walk longer streams. total_items reports the full
    stream length."""
    return render_page(Path(path), offset=offset, count=count, item_type=item_type)


TOOLS: dict[str, Callable[..., object]] = {
    "describe_stages": tool_error_boundary(describe_stages),
    "validate_modem": tool_error_boundary(validate_modem),
    "run_rx": tool_error_boundary(run_rx_tool),
    "run_tx": tool_error_boundary(run_tx_tool),
    "read_stream": tool_error_boundary(read_stream),
}
