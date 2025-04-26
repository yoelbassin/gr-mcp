from itertools import count


def get_unique_id(flowgraph_blocks, base_id=""):
    block_ids = set(b.name for b in flowgraph_blocks)
    for index in count():
        block_id = "{}_{}".format(base_id, index)
        if block_id not in block_ids:
            break
    return block_id
