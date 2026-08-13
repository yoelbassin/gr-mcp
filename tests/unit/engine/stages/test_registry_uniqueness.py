from __future__ import annotations

from typing import Any

import pytest

from marconi.engine.stages import registry
from marconi.engine.stages.base import Stage
from marconi.engine.stages.registry import stage_registry, step_models


def test_every_registered_class_reaches_the_registry() -> None:
    classes = [cls for group in registry._GROUPS for cls in group]
    assert len(stage_registry()) == len(classes)
    assert len(step_models()) == len(classes)


def test_a_duplicate_stage_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # a dict comprehension kept whichever class was listed last and dropped the
    # other from describe_stages, step_models and every path resolution
    first: type[Stage[Any, Any]] = registry._GROUPS[0][0]

    class Clashing(first):  # type: ignore[valid-type, misc]
        pass

    monkeypatch.setattr(registry, "_GROUPS", (*registry._GROUPS, (Clashing,)))
    with pytest.raises(RuntimeError, match=first.name):
        stage_registry()
