"""The curated block vocabulary: compositional primitives plus a few named
compositions (nbfm_rx/nbfm_tx). Dtypes: "c" = complex64 stream, "f" = float32.

Rate-dependent blocks accept an optional `sample_rate` param that defaults to
the pipeline's sample_rate at build time.
"""

from dataclasses import dataclass, field

from marconi.models import BlockSpec, PipelineSpec, ValidationIssue


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    required: bool = False
    default: object = None


@dataclass(frozen=True)
class BlockDef:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: tuple[Param, ...] = field(default_factory=tuple)


_RATE = Param("sample_rate", float)

VOCABULARY: dict[str, BlockDef] = {
    "tone_source": BlockDef(
        (),
        ("c",),
        (
            Param("freq", float, required=True),
            Param("amplitude", float, default=1.0),
            _RATE,
        ),
    ),
    "audio_tone_source": BlockDef(
        (),
        ("f",),
        (
            Param("freq", float, required=True),
            Param("amplitude", float, default=0.5),
            _RATE,
        ),
    ),
    "noise_source": BlockDef(
        (),
        ("c",),
        (Param("amplitude", float, required=True), Param("seed", int, default=0)),
    ),
    "file_source": BlockDef(
        (),
        ("c",),
        (Param("path", str, required=True), Param("repeat", bool, default=False)),
    ),
    "head": BlockDef(("c",), ("c",), (Param("num_samples", int, required=True),)),
    "add": BlockDef(("c", "c"), ("c",)),
    "multiply_const": BlockDef(("c",), ("c",), (Param("value", float, required=True),)),
    "freq_shift": BlockDef(
        ("c",), ("c",), (Param("offset", float, required=True), _RATE)
    ),
    "freq_xlating_lowpass": BlockDef(
        ("c",),
        ("c",),
        (
            Param("decimation", int, required=True),
            Param("center_offset", float, required=True),
            Param("cutoff", float, required=True),
            Param("transition", float, required=True),
            _RATE,
        ),
    ),
    "quadrature_demod": BlockDef(("c",), ("f",), (Param("gain", float, default=1.0),)),
    "rational_resampler_f": BlockDef(
        ("f",),
        ("f",),
        (
            Param("interpolation", int, required=True),
            Param("decimation", int, required=True),
        ),
    ),
    "rational_resampler_c": BlockDef(
        ("c",),
        ("c",),
        (
            Param("interpolation", int, required=True),
            Param("decimation", int, required=True),
        ),
    ),
    "fm_deemphasis": BlockDef(
        ("f",), ("f",), (Param("tau", float, default=75e-6), _RATE)
    ),
    "nbfm_rx": BlockDef(
        ("c",),
        ("f",),
        (
            Param("audio_rate", int, required=True),
            Param("quad_rate", int, required=True),
            Param("tau", float, default=75e-6),
            Param("max_dev", float, default=5e3),
        ),
    ),
    "nbfm_tx": BlockDef(
        ("f",),
        ("c",),
        (
            Param("audio_rate", int, required=True),
            Param("quad_rate", int, required=True),
            Param("tau", float, default=75e-6),
            Param("max_dev", float, default=5e3),
        ),
    ),
    "file_sink": BlockDef(("c",), (), (Param("path", str, required=True),)),
    "wav_sink": BlockDef(
        ("f",),
        (),
        (Param("path", str, required=True), Param("sample_rate", int, required=True)),
    ),
}

_DTYPE_NAMES = {"c": "complex", "f": "float"}


class PipelineValidationError(Exception):
    """Raised when a pipeline fails validation; formats issues for the agent."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        lines = []
        for i in issues:
            where = i.block_id or "<pipeline>"
            f = f".{i.field}" if i.field else ""
            lines.append(f"{where}{f}: {i.message}")
        super().__init__("pipeline validation failed:\n" + "\n".join(lines))


def _check_param_type(value: object, expected: type) -> bool:
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _endpoint_def(by_id: dict[str, str], end: str) -> BlockDef | None:
    """The BlockDef for a connection endpoint, or None if the block id is
    unknown or its declared type isn't in the vocabulary."""
    return VOCABULARY.get(by_id.get(end, ""))


def _check_block_params(
    b: BlockSpec, d: BlockDef, issues: list[ValidationIssue]
) -> None:
    defs = {p.name: p for p in d.params}
    for name in b.params:
        if name not in defs:
            issues.append(
                ValidationIssue(
                    block_id=b.id,
                    field=name,
                    message=f"unknown parameter for {b.type}; "
                    f"accepted: {sorted(defs) or 'none'}",
                )
            )
    for p in d.params:
        if p.required and p.name not in b.params:
            issues.append(
                ValidationIssue(
                    block_id=b.id,
                    field=p.name,
                    message=f"required parameter missing ({p.type.__name__})",
                )
            )
        elif p.name in b.params and not _check_param_type(b.params[p.name], p.type):
            issues.append(
                ValidationIssue(
                    block_id=b.id,
                    field=p.name,
                    message=f"expected {p.type.__name__}, "
                    f"got {type(b.params[p.name]).__name__}",
                )
            )


def _check_blocks(spec: PipelineSpec, issues: list[ValidationIssue]) -> dict[str, str]:
    """Validate each block's type and params; return {block_id: type} for every
    non-duplicate block (including unknown types, so connection checks can tell
    'unknown block' apart from 'unknown type')."""
    by_id: dict[str, str] = {}
    for b in spec.blocks:
        if b.id in by_id:
            issues.append(
                ValidationIssue(block_id=b.id, message=f"duplicate block id '{b.id}'")
            )
            continue
        by_id[b.id] = b.type
        d = VOCABULARY.get(b.type)
        if d is None:
            known = ", ".join(sorted(VOCABULARY))
            issues.append(
                ValidationIssue(
                    block_id=b.id,
                    message=f"unknown block type '{b.type}'; known types: {known}",
                )
            )
            continue
        _check_block_params(b, d, issues)
    return by_id


def _check_connections(
    spec: PipelineSpec, by_id: dict[str, str], issues: list[ValidationIssue]
) -> set[tuple[str, int]]:
    """Validate endpoints, port ranges, double-connected inputs, and dtype
    matches; return the set of (block_id, input_port) that got connected."""
    connected: set[tuple[str, int]] = set()
    for c in spec.connections:
        for end in (c.src_block, c.dst_block):
            if end not in by_id:
                issues.append(
                    ValidationIssue(
                        message=f"connection references unknown block '{end}'"
                    )
                )
        src_d = _endpoint_def(by_id, c.src_block)
        dst_d = _endpoint_def(by_id, c.dst_block)
        if src_d is not None and c.src_port >= len(src_d.outputs):
            issues.append(
                ValidationIssue(
                    block_id=c.src_block,
                    message=f"output port {c.src_port} out of range "
                    f"({len(src_d.outputs)} outputs)",
                )
            )
        if dst_d is not None:
            if c.dst_port >= len(dst_d.inputs):
                issues.append(
                    ValidationIssue(
                        block_id=c.dst_block,
                        message=f"input port {c.dst_port} out of range "
                        f"({len(dst_d.inputs)} inputs)",
                    )
                )
            else:
                key = (c.dst_block, c.dst_port)
                if key in connected:
                    issues.append(
                        ValidationIssue(
                            block_id=c.dst_block,
                            message=f"input port {c.dst_port} connected twice",
                        )
                    )
                connected.add(key)
        if (
            src_d is not None
            and dst_d is not None
            and c.src_port < len(src_d.outputs)
            and c.dst_port < len(dst_d.inputs)
        ):
            out_t, in_t = src_d.outputs[c.src_port], dst_d.inputs[c.dst_port]
            if out_t != in_t:
                issues.append(
                    ValidationIssue(
                        block_id=c.dst_block,
                        message=f"dtype mismatch: {c.src_block} outputs "
                        f"{_DTYPE_NAMES[out_t]} but {c.dst_block} input "
                        f"{c.dst_port} expects {_DTYPE_NAMES[in_t]}",
                    )
                )
    return connected


def _check_inputs_connected(
    spec: PipelineSpec,
    connected: set[tuple[str, int]],
    issues: list[ValidationIssue],
) -> None:
    for b in spec.blocks:
        d = VOCABULARY.get(b.type)
        if d is None:
            continue
        for port in range(len(d.inputs)):
            if (b.id, port) not in connected:
                issues.append(
                    ValidationIssue(
                        block_id=b.id, message=f"input port {port} is not connected"
                    )
                )


def _check_outputs_connected(spec: PipelineSpec, issues: list[ValidationIssue]) -> None:
    fed = {(c.src_block, c.src_port) for c in spec.connections}
    for b in spec.blocks:
        d = VOCABULARY.get(b.type)
        if d is None:
            continue
        for port in range(len(d.outputs)):
            if (b.id, port) not in fed:
                issues.append(
                    ValidationIssue(
                        block_id=b.id, message=f"output port {port} is not connected"
                    )
                )


def validate_pipeline(spec: PipelineSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not spec.blocks:
        return [ValidationIssue(message="pipeline has no blocks")]
    by_id = _check_blocks(spec, issues)
    connected = _check_connections(spec, by_id, issues)
    _check_inputs_connected(spec, connected, issues)
    _check_outputs_connected(spec, issues)
    return issues
