from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from marconi.bits.models import CodecSpec, ParseField
from marconi.core.levels import Level
from marconi.core.models import ValidationIssue
from marconi.core.stages import Stage, validate_path


def validate_codec(
    codec: CodecSpec, registry: Mapping[str, Stage[Any]]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not codec.path:
        return [ValidationIssue(message="codec has no converters")]
    first_conv = codec.path[0].conv
    start = registry[first_conv].from_level if first_conv in registry else Level.BITS
    validate_path(codec.path, registry, start, "codec", issues)
    stages = [registry[s.conv] for s in codec.path if s.conv in registry]
    if stages:
        # Capability comes from the stages' own flags, never from a name-set —
        # a user-registered seeder/slicer advertises itself (issue 08).
        if not any(s.seeds_frames for s in stages):
            issues.append(
                ValidationIssue(
                    message="codec needs a frame seeder (a BITS->FRAMES stage "
                    "declaring seeds_frames, e.g. 'hdlc_deframe')"
                )
            )
        elif not any(s.self_slicing or s.slices_body for s in stages):
            issues.append(
                ValidationIssue(
                    message="codec needs a body slicer (a stage declaring "
                    "self_slicing or slices_body)"
                )
            )
        # parse is OPTIONAL: a deframe+integrity codec of an unknown protocol
        # (the reverse-engineering entry state) is a valid decode to frames.
        for idx, step in enumerate(codec.path):
            if step.conv == "parse":
                fields = step.params.get("fields")
                if isinstance(fields, list):
                    for fi, f in enumerate(fields):
                        try:
                            ParseField.model_validate(f)
                        except ValidationError:
                            issues.append(
                                ValidationIssue(
                                    block_id=f"parse[{idx}]",
                                    field=f"fields[{fi}]",
                                    message="each parse field needs a string 'name' "
                                    "and an int 'bits'",
                                )
                            )
    return issues
