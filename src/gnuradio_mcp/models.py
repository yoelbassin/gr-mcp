from pydantic import BaseModel
from gnuradio.grc.core.blocks.block import Block


class BlockModel(BaseModel):
    label: str
    key: str

    @classmethod
    def from_block(cls, block: Block) -> "BlockModel":
        return cls(label=block.label, key=block.key)