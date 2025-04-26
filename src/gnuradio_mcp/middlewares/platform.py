from typing import List
from gnuradio.grc.core.platform import Platform
from pydantic import BaseModel

from gnuradio.grc.core.blocks.block import Block

class BlockModel(BaseModel):
    label: str
    key: str

    @classmethod
    def from_block(cls, block: Block) -> "BlockModel":
        return cls(label=block.label, key=block.key)

class PlatformMiddleware:
    def __init__(self, platform: Platform):
        self._platform = platform

    @property
    def blocks(self) -> List[BlockModel]:
        return [BlockModel.from_block(block) for block in self._platform.blocks.values()]
