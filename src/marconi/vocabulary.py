"""The curated block vocabulary: compositional primitives plus a few named
compositions (nbfm_rx/nbfm_tx). Dtypes: "c" = complex64 stream, "f" = float32.

Rate-dependent blocks accept an optional `sample_rate` param that defaults to
the pipeline's sample_rate at build time.
"""

from dataclasses import dataclass, field

from marconi.models import PipelineSpec, ValidationIssue


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


def validate_pipeline(spec: PipelineSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
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

    connected_inputs: set[tuple[str, int]] = set()
    for c in spec.connections:
        for end in (c.src_block, c.dst_block):
            if end not in by_id:
                issues.append(
                    ValidationIssue(
                        message=f"connection references unknown block '{end}'"
                    )
                )
        if c.src_block in by_id and by_id[c.src_block] in VOCABULARY:
            d = VOCABULARY[by_id[c.src_block]]
            if c.src_port >= len(d.outputs):
                issues.append(
                    ValidationIssue(
                        block_id=c.src_block,
                        message=f"output port {c.src_port} out of range "
                        f"({len(d.outputs)} outputs)",
                    )
                )
        if c.dst_block in by_id and by_id[c.dst_block] in VOCABULARY:
            d = VOCABULARY[by_id[c.dst_block]]
            if c.dst_port >= len(d.inputs):
                issues.append(
                    ValidationIssue(
                        block_id=c.dst_block,
                        message=f"input port {c.dst_port} out of range "
                        f"({len(d.inputs)} inputs)",
                    )
                )
            else:
                key = (c.dst_block, c.dst_port)
                if key in connected_inputs:
                    issues.append(
                        ValidationIssue(
                            block_id=c.dst_block,
                            message=f"input port {c.dst_port} connected twice",
                        )
                    )
                connected_inputs.add(key)
        if (
            c.src_block in by_id
            and c.dst_block in by_id
            and by_id[c.src_block] in VOCABULARY
            and by_id[c.dst_block] in VOCABULARY
        ):
            src_d = VOCABULARY[by_id[c.src_block]]
            dst_d = VOCABULARY[by_id[c.dst_block]]
            if c.src_port < len(src_d.outputs) and c.dst_port < len(dst_d.inputs):
                out_t = src_d.outputs[c.src_port]
                in_t = dst_d.inputs[c.dst_port]
                if out_t != in_t:
                    issues.append(
                        ValidationIssue(
                            block_id=c.dst_block,
                            message=f"dtype mismatch: {c.src_block} outputs "
                            f"{_DTYPE_NAMES[out_t]} but {c.dst_block} input "
                            f"{c.dst_port} expects {_DTYPE_NAMES[in_t]}",
                        )
                    )

    for b in spec.blocks:
        d = VOCABULARY.get(b.type)
        if d is None:
            continue
        for port in range(len(d.inputs)):
            if (b.id, port) not in connected_inputs:
                issues.append(
                    ValidationIssue(
                        block_id=b.id,
                        message=f"input port {port} is not connected",
                    )
                )

    return issues
