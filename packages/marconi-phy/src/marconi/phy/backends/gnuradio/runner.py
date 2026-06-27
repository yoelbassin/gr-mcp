from __future__ import annotations

from typing import Any

from marconi.phy.backends.base import Backend
from marconi.phy.backends.gnuradio.build import build_top_block
from marconi.phy.ir import GrPipeline


class GnuRadioBackend(Backend):
    @property
    def name(self) -> str:
        return "gnuradio-3.10"

    def instantiate(self, pipeline: GrPipeline) -> Any:
        return build_top_block(pipeline)
