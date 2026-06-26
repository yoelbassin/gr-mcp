from marconi.core.levels import Level, adjacent


def test_main_ladder_is_adjacent_downward() -> None:
    assert adjacent(Level.IQ, Level.SYMBOLS)
    assert adjacent(Level.SYMBOLS, Level.BITS)
    assert adjacent(Level.BITS, Level.FRAMES)
    assert adjacent(Level.FRAMES, Level.MESSAGES)


def test_audio_is_a_first_class_branch_off_iq() -> None:
    assert adjacent(Level.IQ, Level.AUDIO)  # resolves C3: AUDIO routable


def test_same_level_is_adjacent_for_conditioning_stages() -> None:
    assert adjacent(Level.IQ, Level.IQ)
    assert adjacent(Level.BITS, Level.BITS)


def test_non_adjacent_and_upward_are_rejected() -> None:
    assert not adjacent(Level.IQ, Level.BITS)  # skips a rung
    assert not adjacent(Level.BITS, Level.SYMBOLS)  # upward (RX moves down)
    assert not adjacent(Level.AUDIO, Level.SYMBOLS)
