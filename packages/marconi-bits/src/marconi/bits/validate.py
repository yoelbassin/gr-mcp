from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from marconi.bits.models import CodecSpec, ParseField
from marconi.core.levels import Level
from marconi.core.models import ValidationIssue
from marconi.core.stages import Stage, validate_path

# These sets grow as stages are added (access_code/segment seeders;
# fixed_frame/var_length_frame slicers) — slice 1 ships only hdlc_deframe.
_SEEDERS = {"hdlc_deframe", "segment"}
_SELF_SLICING = {"hdlc_deframe"}
_BODY_SLICERS: set[str] = {"fixed_frame"}


def validate_codec(
    codec: CodecSpec, registry: Mapping[str, Stage[Any]]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not codec.path:
        return [ValidationIssue(message="codec has no converters")]
    validate_path(codec.path, registry, Level.BITS, "codec", issues)
    present = {s.conv for s in codec.path if s.conv in registry}
    if present:
        if not (present & _SEEDERS):
            issues.append(
                ValidationIssue(
                    message="codec needs a frame seeder (e.g. 'hdlc_deframe')"
                )
            )
        if not (present & _SELF_SLICING) and not (present & _BODY_SLICERS):
            issues.append(ValidationIssue(message="codec needs a body slicer"))
        if "parse" not in present:
            issues.append(
                ValidationIssue(
                    message="codec is missing the required 'parse' converter"
                )
            )
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
