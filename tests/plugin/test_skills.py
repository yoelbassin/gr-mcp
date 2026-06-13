from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"
EXPECTED = {
    "survey-spectrum",
    "build-receiver",
    "debug-no-signal",
    "simulate-scene",
    "tx-experiment",
    "escape-hatch",
}


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} missing frontmatter"
    _, fm, _body = text.split("---\n", 2)
    return yaml.safe_load(fm)


def test_every_present_skill_has_valid_frontmatter():
    found = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert found, "no skills found"
    for name in found:
        fm = _frontmatter(SKILLS / name / "SKILL.md")
        assert fm.get("name") == name
        assert isinstance(fm.get("description"), str) and len(fm["description"]) > 20


def test_all_expected_skills_present():
    found = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert found == EXPECTED
