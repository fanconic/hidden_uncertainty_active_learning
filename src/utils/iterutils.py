# Taken from : https://github.com/ElementAI/baal/blob/a9cc0034c40d0541234a3c27ff5ccbd97278bcb3/baal/utils/iterutils.py#L4

from collections.abc import Sequence


def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    elif isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)
