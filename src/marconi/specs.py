from pathlib import Path

import yaml

from marconi.models import PipelineSpec, SceneSpec


def save_pipeline(spec: PipelineSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_pipeline(path: Path | str) -> PipelineSpec:
    return PipelineSpec.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def save_scene(scene: SceneSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(scene.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_scene(path: Path | str) -> SceneSpec:
    return SceneSpec.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )
