# Marconi Roadmap

> **The one-line version:** make Marconi the tool an RF engineer reaches for *first*.

## North star

Marconi is **"Claude Code for RF"** — an LLM-driven toolkit built to become *indispensable*
to working RF engineers and researchers. The ambition is not a clever demo; it is to be
**the default way RF work gets done**: the assistant an engineer trusts to survey a band,
reason about a signal, build and run the processing, analyze the result, and iterate — in
the language they already think in.

Everything below is justified by a single question:

> **Does it move Marconi closer to being the tool an RF engineer reaches for first?**

If a capability doesn't serve that, it doesn't belong here — no matter how impressive the
demo. Marconi is delivered first as a **Claude Code plugin**; the engine-agnostic core
(`src/marconi/`) is the portability anchor, so when the time is right a standalone product
reuses it unchanged (see [the vision](#-beyond-v12--the-vision)).

## What "indispensable" requires

An RF engineer's real work runs an arc, and Marconi has to be genuinely useful across all of
it — not just one slice. Here is the arc, and an honest account of where Marconi stands today:

| Stage of the work | What it means | Today |
|---|---|---|
| **Survey** | See what's on the air or in a recording — power spectrum, find signals, detect bursts | ✅ |
| **Understand** | Characterize a signal — center frequency, bandwidth, power, shape | ✅ basics · ⚠️ no modulation recognition yet |
| **Build** | Express the processing as a pipeline — channelize, filter, demodulate — validated before it runs | ✅ analog |
| **Run** | Execute that pipeline on samples or a device | ✅ (GNU Radio backend) |
| **Analyze & render** | Pull out results and see them — spectrogram, PSD, constellation | ✅ |
| **Simulate** | Generate scenes with known ground truth to prototype and test against | ✅ |
| **Transmit** | Close the loop — transmit and receive back | ✅ simulation, gated |
| **Decode** | Turn a digital signal into bits and messages | ❌ → **v1.1** |
| **Live hardware** | Do all of the above on real air, with real radios | ❌ → **v1.2** |
| **Bench instruments** | Drive spectrum analyzers and signal generators | ❌ → vision |

Marconi already does the **analog, simulated loop** end-to-end and does it well. Two gaps
stand between that and an engineer's daily driver:

1. **It can't decode.** The moment a signal is digital, Marconi can demodulate the analog
   layer but not recover the message. This is the largest slice of real RF work it can't yet touch.
2. **It can't touch real hardware.** Everything works against simulation; the engineer's
   actual radio is still on the other side of the glass.

Those two gaps set the near-term sequence. Everything else is vision.

## Strategy: the wedge

We earn "indispensable" the way the best tools do — narrow first, then trusted, then wide.

- **Narrow but genuinely useful.** Be excellent at a real, complete workflow before being
  mediocre at many. v1.0 picks the analog/FM + simulation loop and finishes it.
- **Earn trust, then widen.** Capability grows from *real usage* — the gaps engineers
  actually hit — not from demo pressure.
- **Method, not recipes.** Skills encode transferable RF *judgment*, not demo scripts. The
  review test for any skill: *"could an RF engineer follow this for AM or SSB, not just FM?"*
  The product's value is judgment; features are just what the judgment operates on.
- **Core as portability anchor.** The engine-agnostic core stays clean and engine-free so it
  can outlive both GNU Radio and the plugin form factor.
- **Open formats.** Captures are SigMF, pipelines/scenes are YAML, flowgraphs export to
  `.grc` — your work is usable without Marconi, and outlives it.

## Milestones — each one a moment an RF engineer can now…

### ✅ v1.0 — Simulation foundation *(shipped)*

> *"…prototype the full RF loop against known ground truth, without any hardware."*

The engine-agnostic core, the GNU Radio execution backend, the MCP server, the Claude Code
plugin, and six method-first skills. Survey a simulated band, find the signals, build an NBFM
receiver, recover the audio — and run a closed-loop TX/RX experiment — entirely in
simulation. See [`docs/superpowers/specs/`](docs/superpowers/specs/) and
[`docs/superpowers/plans/`](docs/superpowers/plans/).

### 🎯 v1.1 — Decode digital signals *(next)*

> *"…point Marconi at a recording of a digital transmission and get messages back."*

**Why this is next:** decoding is the single biggest slice of real RF work Marconi can't yet
touch, and — unlike hardware — it delivers value on any machine, no radio required. It moves
Marconi from "demodulates analog" to "tells you what the signal *says*."

**The bet: general primitives, not per-protocol decoders.** OOK / PPM / FSK → symbols →
bits, with framing and CRC helpers, added to the curated vocabulary. **ADS-B** (1090 MHz
aircraft messages) is the worked example that proves it end-to-end — but the design test is
*"could the same primitives decode POCSAG or AIS?"* A per-protocol bolt-on would be a demo;
a primitive layer is a capability. Gets its own brainstorm → spec → plan before any code.

### 📡 v1.2 — Work on live air *(hardware receive)*

> *"…run the exact same workflow on a $30 RTL-SDR instead of a simulation."*

The same skills, the same pipelines — `rtl0` in place of `sim0`. Hardware receive arrives as
GNU Radio in-tree blocks (`gr-soapy`, `gr-uhd`) behind the *existing* device/backend
interface, ideally with **zero skill changes**. This is the milestone that proves the entire
device abstraction against the messiness of real RF: tune to a live FM station, build the
receiver, play real audio.

### 🔭 Beyond v1.2 — the vision

Ordered by how directly each serves the north star, not by how easy it is:

- **Transmit on real hardware** + experiment automation — closed-loop BER-vs-SNR sweeps with
  a written report. (TX stays gated by `CONFIRM_TX` — a guard against agent mistakes like a
  wrong frequency or device, *not* a licensing control; licensing is the operator's
  responsibility.)
- **Understand signals it has never seen** — modulation recognition and auto-classification,
  so an engineer can ask *"what is this?"* and not only *"demodulate this FM."*
- **The standalone product** — graduate from a Claude Code plugin to a first-class RF
  platform with its own agent UI, reusing the core unchanged. The architecture already
  protects this path.
- **The whole bench** — T&M / SCPI instrument control (spectrum analyzers, signal
  generators), so Marconi spans the bench, not just SDRs.
- **Round-trip with GNU Radio** — `.grc` *import*, non-blocking runs with stop, and mid-run
  parameter tweaks.

## How we decide what's next

The escape-hatch skill exists to be used: when the vocabulary genuinely can't express a task,
the agent drops to Python against the core API and **reports the gap**. Those gap reports —
together with what real users actually reach for — *are* the operations roadmap. New
modulations, blocks, and operations get added deliberately, when usage demands them, never
under demo pressure.

## Cross-cutting (ongoing)

- **Quality as a release gate** — agent-transcript skill evals (scored scenarios with ground
  truth) run before each release, with an anti-overfit case to guard against demo-specific skills.
- **Vocabulary growth** — AM/SSB and beyond, driven by the gap reports above.
- **Polish backlog** — small follow-ups carried from v1.0: TX-error classification scope, a
  concurrency-safe run counter for future non-blocking runs, and `export_grc`/`save_pipeline`
  delegation.

---

*This roadmap supersedes the tentative phasing in the v1.0 design doc, which had sequenced
hardware ahead of digital decode. Directions and ordering will keep changing as we learn from
real use — the **north star** will not.*
