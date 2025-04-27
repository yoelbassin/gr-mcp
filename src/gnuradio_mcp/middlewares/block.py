from __future__ import annotations

from typing import Any, Dict

from gnuradio.grc.core.blocks.block import Block

from gnuradio_mcp.middlewares.base import ElementMiddleware
from gnuradio_mcp.models import SINK, SOURCE, ParamModel, PortModel


class BlockMiddleware(ElementMiddleware):
    def __init__(self, block: Block):
        super().__init__(block)
        self._block = self._element

    @property
    def name(self) -> str:
        return self._block.name

    @name.setter
    def name(self, name: str):
        self._block.params["id"].set_value(name)

    def set_param(self, param_name: str, param_value: Any):
        self._block.params[param_name].set_value(param_value)

    def set_params(self, params: Dict[str, Any]):
        for param_name, param_value in params.items():
            self.set_param(param_name, param_value)

    # TODO: Check if rewrite is needed
    @property
    def params(self) -> list[ParamModel]:
        return [ParamModel.from_param(param) for param in self._block.params.values()]

    @property
    def sinks(self) -> list[PortModel]:
        self._rewrite()
        ports = []
        for port in self._block.sinks:
            port_model = PortModel.from_port(port, SINK)
            if not port_model.hidden:
                ports.append(port_model)
        return ports

    @property
    def sources(self) -> list[PortModel]:
        self._rewrite()
        ports = []
        for port in self._block.sources:
            port_model = PortModel.from_port(port, SOURCE)
            if not port_model.hidden:
                ports.append(port_model)
        return ports
