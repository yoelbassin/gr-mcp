from pathlib import Path

from pydantic import BaseModel

from marconi.engine.io.specs import load_yaml_spec, save_yaml_spec


class _Spec(BaseModel):
    name: str
    n: int


def test_yaml_spec_round_trips(tmp_path: Path) -> None:
    p = save_yaml_spec(_Spec(name="m", n=3), tmp_path / "m.yaml")
    assert load_yaml_spec(_Spec, p) == _Spec(name="m", n=3)
