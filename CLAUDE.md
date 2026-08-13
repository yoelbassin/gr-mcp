# CLAUDE.md
## Marconi — Product Brief

Marconi is an LLM-driven assistant for radio-frequency work — **"Claude Code for RF."**
Operators speak in natural language and Marconi does the radio work on their behalf:
surveying a band, identifying an unknown signal, decoding a transmission, or
constructing one. Its value is expert RF judgment delivered through conversation,
removing the slow specialist loop of hand-building flowgraphs, consulting references,
and writing single-use scripts.

**Intended users** — practitioners who work with the signals already in their
environment, not engineers designing radios of their own:

- Makers and amateur-radio operators — receiving/decoding nearby transmissions and
  extending their own equipment.
- Security and reverse-engineering researchers (authorized contexts) — characterizing
  undocumented protocols and exercising devices under test.
- Educators and students — generating signals with known ground truth, then decoding
  and measuring them.

**Two-sided workflow**, each side informing the other:

- **Understand** — measure a signal's parameters, classify its modulation, and turn IQ
  into FEC-corrected bits; the agent composes those bits with its own framing, CRC, and
  field-parsing to read messages and reverse-engineer an unknown protocol's structure.
- **Create** — encode messages into a signal, generate test waveforms, and transmit —
  sim-first, and on hardware only behind an explicit safety confirmation.

Marconi itself is a radio: it transforms between IQ and FEC-corrected bits — sync,
demod, symbol decisions, descrambling, deinterleaving, FEC. Framing, CRC checks, field
parsing, and messages are protocol-datasheet work, not the product's. Reverse-engineering
a protocol from Marconi's bits is exactly what enables recreating and transmitting it —
closing the loop between the two sides.

**Success:** a user points Marconi at a band or a recording, asks "what is this, and
what is it transmitting?" in plain language, and gets a substantive, trustworthy
answer — and, when desired, builds and transmits a signal of their own.

## Coding Standards and Guidelines

Living document — the rules every commit holds to. When a rule and
the code disagree, one is a bug — fix it, don't document the drift.
Marconi is protocol agnostic; it does not implement any protocols itself -
it builds the tools that allow users to implement protocols.
Specific protocols don't live inside the Marconi ecosystem itself:
not in the source code, docs, agent skills or examples.
The dividing test: does this need DSP judgment, or a protocol datasheet? DSP judgment —
sync, demod, symbol decisions, descrambling, deinterleaving, FEC — is a phy stage;
datasheet knowledge — frame formats, CRC parameters, field layouts, message semantics —
belongs in tests or the driving agent, never the product.
If a new protocol needs to be added or checked for support, we can
do it as a new test, constructing it using the Marconi API.
- **Don't re-implement the wheel, base on existing implementations.**

When reasoning, brainstorming and planning, always choose whats best long term, with
the product and users in mind. Don't be afraid to do a bigger refactor, a bit more work,
if it will benefit the product in the long run. Question every design decision, but alwayes
think it through.


### Typing

- **Every signature is fully annotated** — arguments and return. No implicit `Any`, no
  untyped public functions.
- **Typed models over loose containers.** A structured payload is a `pydantic` model or
  frozen `dataclass`, not a bare `dict`/`tuple`. If validation reduces to a hand-rolled
  `isinstance` ladder, the data wanted a model.
- **`mypy` is green** via `uv run mypy .` (source and tests); the pre-commit hook runs
  against the installed workspace so the gate and the CLI never disagree.
- **Every distributed package ships `py.typed`** (PEP 561).

### Functions, classes, modules

- **Short, single-responsibility functions.**
- **Cohesive classes own their data and behavior.**


### Styling
- Prefer a `{key: handler}` dispatch table over a long `if/elif` chain.

### Comments & docstrings

Code must explain itself — through better names, named constants, and extracted
helpers. Write **almost no comments and almost no docstrings**. The enemy is **drift**:
a note rots when the thing it describes changes but the note doesn't, so a fact's home
is wherever it's *least likely to drift relative to its source of truth*:

- **Code narration** ("loop twice because…", "returns early when…") — source of truth is
  the code itself; the note just duplicates it. **Delete.** Same for ASCII banners and
  docstrings that echo the signature.
- **A measured constant's provenance** — what was measured, on what, and what the
  number would break if moved (`_SOFT_POSITIVE = 2.0` and the SNR ladder behind it,
  `_RISE_RATIO`, `_DOMINANCE_MARGIN`, every threshold in `quality.py`). **Keep, next
  to the constant.** This does not drift the way narration does: the note's source of
  truth is an experiment that already happened, and the code cannot restate it — a
  reader who deletes it has no way to re-derive the number short of re-running the
  measurement. It is also the note most likely to stop someone "tidying" a calibrated
  threshold. The rule this exception encodes: **write the why where the why cannot be
  read off the code, and nowhere else.**
- **A refuted hypothesis** — "X looks like the fix here and is not, because measured Y".
  **Keep.** It is the only thing standing between the next reader and a re-litigated
  dead end.

Per-protocol values (CRC params, bit order, charset) don't live in the Marconi ecosystem.

**Exception to all of the above:** once an MCP surface exists, its `@tool`-decorated
docstrings are the tool descriptions surfaced to the LLM agent — functional product,
never touched. Because they are product, they are also TESTED: a composition or a
field a docstring names must be exercised by
`tests/unit/engine/stages/test_documented_compositions.py` or an equivalent gate. A
docstring claim with nothing executing it is an untested feature, and three shipped
false (a stage description advertising a path the compiler rejected, a cache bound the
eviction did not deliver, a resolution that described one of two branches).

### Errors

One classifier in `marconi/errors.py`; each package registers its exception types for
stable MCP `[code]` prefixes. Once an MCP surface exists, every MCP tool is wrapped
with `@tool_error_boundary`.

### Tests

- **Test real behavior, not mocks** — round-trip BER-0 sweeps through the real engine,
  AWGN introduction, real CRCs, malformed input. The test tree is kind-first: `tests/unit`
  (one component under test — a minimal compiled chain may be the vehicle; source tree
  mirrored inside), `tests/integration` (multi-stage behavior: round-trips and cross-stage
  contracts through the real engine on synthetic signals), `tests/e2e` (gates against
  external ground truth: off-air captures and sim protocol gates with exact oracles; an
  off-air-derived literal alone does not make a test e2e).
- **Green at every commit** — the full `pytest` suite passes before work is done.

### Tooling

`isort` (black profile) · `black` · `flake8` (max line 88, `E203`/`W503` ignored) ·
`mypy` (`strict`; the only exemptions are the stubless external packages and the
GNU Radio subclassing seam, both named per-module in `pyproject.toml`) ·
`pytest` — all under pre-commit. The commit hook runs `tests/unit` and
`tests/integration`; `tests/e2e` needs off-air assets and hardware, so it gates
the branch, not the keystroke — run the full suite before calling work done.
This file is a living document edited in place, never a dated copy.
