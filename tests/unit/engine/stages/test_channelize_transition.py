"""The anti-alias transition must land the stopband edge ON the decimated
output's Nyquist: tying it to the passband (bandwidth/2) left energy that
folds in attenuated by only 22.5 dB at the guard's own legal boundary
(measured: a tone outside the channel survived decim=4 and became the PEAK
of the channelized spectrum). 22 dB SIR silently destroys QAM."""

from __future__ import annotations

from marconi.engine.stages.conditioning import _transition_hz


def test_stopband_edge_lands_on_the_output_nyquist() -> None:
    rate, decim = 1_000_000.0, 4
    for bw in (200_000.0, 125_000.0, 50_000.0):
        transition = _transition_hz(bw, rate, decim)
        stopband_edge = bw / 2.0 + transition
        assert stopband_edge <= rate / (2.0 * decim) * 1.001, (bw, transition)


def test_taps_floor_still_bounds_the_filter() -> None:
    # at bandwidth -> output-rate the exact-Nyquist transition collapses to
    # 0 and the taps budget must take over (rate/transition sizes the FIR);
    # the residual fold-in there is bounded by the floor, not eliminated
    rate, decim = 1_000_000.0, 4
    for bw in (249_900.0, 250_000.0):
        assert _transition_hz(bw, rate, decim) >= rate / 4096.0
