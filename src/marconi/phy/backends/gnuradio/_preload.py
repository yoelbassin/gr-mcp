"""Imported by the forkserver daemon (never the parent) so each forked run
starts from a process that already has gnuradio + the worker module loaded."""

from gnuradio import gr  # noqa: F401
from gnuradio import analog, blocks, digital  # noqa: F401
from gnuradio import filter as gr_filter  # noqa: F401

import marconi.phy.backends.gnuradio.worker  # noqa: F401
