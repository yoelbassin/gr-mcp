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

- **Understand** — measure a signal's parameters, classify its modulation, decode it
  into messages, reverse-engineer an unknown protocol down to its structure.
- **Create** — encode messages into a signal, generate test waveforms, and transmit —
  sim-first, and on hardware only behind an explicit safety confirmation.

Reverse-engineering a protocol is exactly what enables recreating and transmitting it —
closing the loop between the two sides.

**Success:** a user points Marconi at a band or a recording, asks "what is this, and
what is it transmitting?" in plain language, and gets a substantive, trustworthy
answer — and, when desired, builds and transmits a signal of their own.

## Coding Standards and Guidelines

Living document — the rules every commit holds to. When a rule and
the code disagree, one is a bug — fix it, don't document the drift.
Marconi is protocl agnostic; it does not implement any protocols itself -
it builds the tools that allow users to implement protocols.
Specific protocols don't live inside the Marconi ecosystem itself:
not in the source code, docs, agent skills or examples.
If a new protocol needs to to be added or checked for support, we can
do it as a new test, constructing it using the Marconi API.
- **Don't re-implement the wheel, base on existing implementations.**



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

Per-protocol values (CRC params, bit order, charset) don't live in the Marconi ecosystem.

**Exception to all of the above:** the
`@tool`-decorated docstrings in `mcp/tools.py` are the MCP tool descriptions surfaced to
the LLM agent — functional product, never touched.

### Errors

One classifier in `core/errors.py`; each package registers its exception types for
stable MCP `[code]` prefixes. Every MCP tool is wrapped with `@tool_error_boundary`.

### Tests

- **Test real behavior, not mocks** — round-trip BER-0 sweeps through the real engine,
  AWGN introduction, real CRCs, malformed input. The test tree mirrors the source tree,
  with seperation into units, integration and e2e.
- **Green at every commit** — the full `pytest` suite passes before work is done.

### Tooling

`isort` (black profile) · `black` · `flake8` (max line 88, `E203`/`W503` ignored) ·
`mypy` · `pytest`, all under pre-commit. The docs (`PRODUCT`, `SPEC`, `CONTEXT`, this
file) are living files edited in place, never dated copies.
