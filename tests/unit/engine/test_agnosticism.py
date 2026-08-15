"""CLAUDE.md's product-defining rule: protocols do not live in the Marconi
ecosystem — not in source, docs, agent skills, or examples. Issue 04 removed
LoRa's datasheet and AIS's preamble from production; this guard keeps them out.

A protocol's identity leaks three ways: by NAME (an identifier, string, or
comment that only makes sense for one protocol), by TABLE (its FEC parity /
sync words / charset as a literal), or by FORMULA (arithmetic bit-identical to
one vendor's datasheet). This test enforces the first two structurally across
every distributed package; the third is covered by keeping the frame-length
math parameterized (tests/unit/helpers/test_coding.py proves LoRa's values live
in tests)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from helpers._paths import SRC_MARCONI as _SRC

# Tokens that name a specific protocol/vendor/capture. Word-boundaried and
# case-insensitive. "hdlc" is NOT here — HDLC framing is a generic mechanism;
# its AIS-specific training sequence is now a caller parameter.
_PROTOCOL_TOKENS = [
    r"lora",
    r"semtech",
    r"sx12[0-9]{2}",
    r"ldro",
    r"netid",
    r"ax\.?25",
    r"\bais\b",
    r"\bdab\b",
    r"welle",
    r"fingerpring",  # the IQ_2 capture's payload string
    r"iq_?\d",  # capture names IQ_1, IQ_2, IQ_12
    r"ads-?b",
    r"acars",
    r"flinders",
    r"pocsag",
    r"bluetooth",
    r"\bble\b",
    r"nordic",
    r"nrf5[0-9]",
    r"\bdmr\b",
    r"mototrbo",
    r"\bbptc\b",
    r"\bcsbk\b",
    r"capacity\s*plus",
    r"\bdsd\b",
    r"\bdrm\b",
    r"\bmode\s*b\b",
    r"deutsche\s*welle",
    r"\bdw\s*drm\b",
    r"journaline",
    r"1024-?phase",
    r"gain\s*reference",
    r"168,\s*255,\s*161",  # DRM scattered-pilot Z-row literal
    r"zigbee",
    r"802\.?15\.?4",
    r"\brds\b",
    r"\bwm-?bus\b",
    r"\bm-?bus\b",
    # Protocols with no capture in the suite yet. The list is hand-maintained,
    # so its coverage is whatever someone remembered to type — these are the
    # names a new vertical is most likely to arrive under, added ahead of the
    # leak rather than after it.
    r"\bp25\b",
    r"\btetra\b",
    r"\bgsm\b",
    r"\blte\b",
    r"\bmeshtastic\b",
    r"\btpms\b",
    r"\biridium\b",
    r"\binmarsat\b",
    r"\bapco\b",
    r"\bsigfox\b",
    r"\bz-?wave\b",
    r"\b802\.?11\b",
]
_PATTERN = re.compile("|".join(_PROTOCOL_TOKENS), re.IGNORECASE)

# Coding-theory terms that contain no protocol identity (mathematicians, not
# protocols): a bare-word allowlist checked before flagging.
_ALLOWED = re.compile(r"hamming bound|gray|viterbi|trellis", re.IGNORECASE)


def _src_files() -> list[Path]:
    files = [
        p
        for p in _SRC.rglob("*.py")
        if "__pycache__" not in p.parts and "/tests/" not in str(p)
    ]
    assert files, "no package source found — path wrong?"
    return files


def test_no_protocol_name_in_production_source() -> None:
    offenders: list[str] = []
    for path in _src_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = _ALLOWED.sub("", line)
            m = _PATTERN.search(stripped)
            if m:
                rel = path.relative_to(_SRC)
                offenders.append(f"{rel}:{lineno}: {m.group(0)!r} in {line.strip()!r}")
    assert not offenders, "protocol names in production:\n" + "\n".join(offenders)


def test_css_coding_carries_no_parity_table() -> None:
    # The Hamming parity matrices are the protocol's FEC definition; they must
    # arrive as a parameter, never as a module constant.
    coding = _SRC / "engine/coding/primitives.py"
    src = coding.read_text()
    assert "HAMMING_PARITY" not in src
    # no literal list-of-lists (a parity matrix) baked in
    assert not re.search(r"\[\s*\[\s*[01]\s*,", src), "a parity table is inlined"


def test_coding_primitives_surface_is_minimal() -> None:
    # The product keeps only what its ops consume: gray codes, the Hamming
    # bound, and the syndrome-LUT correction math. The explicit-header frame
    # algebra, diagonal deinterleaver, and symbol demap are datasheet-shaped
    # and live in tests/helpers/blockmath.py; this pins the surface so none
    # can quietly return to src/.
    from marconi.engine.coding import primitives

    public = {n for n in dir(primitives) if not n.startswith("_")}
    assert public == {
        "annotations",
        "can_correct",
        "effective_t",
        "flip_table",
        "gray_decode",
        "gray_encode",
        "syndrome_key",
        "syndrome_table",
        "MAX_SYNDROME_PATTERNS",
    }


def _frame_geometry_files() -> list[Path]:
    """Where a frame's or a codeword's size would be written down if it ever
    got written down: the frame/codeword algebra and the compiler that checks
    a codeword against the wire carrying it."""
    files = [_SRC / "engine/modulation/coding/stages.py"]
    files += sorted((_SRC / "engine/compile").glob("*.py"))
    missing = [p for p in files if not p.exists()]
    assert not missing, f"frame-geometry surface moved: {missing}"
    return files


def test_the_frame_geometry_surface_names_no_concrete_sizes() -> None:
    """A four-digit size on THIS surface can only have come from one protocol's
    frame. The tiling rule's rationale once read "an off-air lane hands the tail
    12384 LLRs holding four 3096-LLR codewords" — a rule that is protocol
    agnostic, justified by numbers that are not. Those figures belong to the
    test that gates the rule (tests/unit/engine/compile/test_frame_contract.py),
    which is free to hold them.

    Scoped deliberately, not swept tree-wide: src carries ~50 four-digit numbers
    in comments and every one is the measured-constant provenance CLAUDE.md
    protects (poll cadences, Viterbi state tables, page caps), so a tree-wide
    numeric gate is all false positive. Nothing on this surface has any reason
    to name a size."""
    offenders: list[str] = []
    for path in _frame_geometry_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = re.search(r"\b\d{4,}\b", line)
            if m:
                rel = path.relative_to(_SRC)
                offenders.append(f"{rel}:{lineno}: {m.group(0)} in {line.strip()!r}")
    assert not offenders, (
        "concrete frame/codeword sizes on the frame-geometry surface:\n"
        + "\n".join(offenders)
        + "\nThe rule is agnostic; move the figure to the test that gates it."
    )


@pytest.mark.parametrize(
    "path",
    [
        "engine/backends/gnuradio/embedded/chirp.py",
        "engine/coding/stages_symbols.py",
        "engine/coding/primitives.py",
        "engine/modulation/css/stages.py",
    ],
)
def test_no_inline_sfd_or_demap_constants(path: str) -> None:
    # The 2.25 SFD, the reduced-rate //4, and the -1 demap offset were LoRa's;
    # each is now a parameter. Guard against a literal creeping back in.
    src = (_SRC / path).read_text()
    assert "2.25" not in src, "the 2.25 SFD literal must be a parameter"
    assert "// 4" not in src and "//4" not in src, "the //4 demap must derive"
