from pathlib import Path

from marconi.backends import get_backend
from marconi.models import PipelineSpec, RunResult
from marconi.specs import save_pipeline
from marconi.vocabulary import PipelineValidationError, validate_pipeline
from marconi.workspace import Workspace


def run_pipeline(
    spec: PipelineSpec, timeout: float = 30.0, backend: str = "gnuradio"
) -> RunResult:
    """Validate, then execute. Validation problems raise
    PipelineValidationError with per-block, per-field messages."""
    issues = validate_pipeline(spec)
    if issues:
        raise PipelineValidationError(issues)
    return get_backend(backend).run_pipeline(spec, timeout=timeout)


def save_pipeline_to_workspace(spec: PipelineSpec, workspace: Workspace) -> Path:
    return save_pipeline(spec, workspace.new_pipeline_path(spec.name))
