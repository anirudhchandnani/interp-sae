from contextlib import contextmanager

import torch


def _blocks(model):
    return model.transformer.h


@contextmanager
def steer(model, layer, direction, coef):
    block = _blocks(model)[layer]

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden + coef * direction
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted

    handle = block.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def project(activations, direction):
    unit = direction / direction.norm()
    return activations @ unit
