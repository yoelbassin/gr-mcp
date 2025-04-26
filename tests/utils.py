from gnuradio_mcp.models import BlockModel

def get_block_keys(platform):
    return list(platform.blocks.keys())

def add_block(flowgraph_middleware, block_key, block_name):
    flowgraph_middleware.add_block(block_key, block_name)

def remove_block(flowgraph_middleware, block_name):
    flowgraph_middleware._flowgraph.remove_element(
        flowgraph_middleware._flowgraph.get_block(block_name)
    )

def get_current_blocks(flowgraph_middleware):
    return [
        BlockModel(key=block.key, label=block.label)
        for block in flowgraph_middleware._flowgraph.blocks
    ] 