from contextlib import contextmanager

import torch


def _blocks(model):
    return model.transformer.h


@contextmanager
def steer(model, layer, direction, coef):
    block = _blocks(model)[layer]
    state = {"step": 0}
    callable_coef = callable(coef)

    def hook(module, args, output):
        value = coef(state["step"]) if callable_coef else coef
        state["step"] += 1
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden + value * direction
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted

    handle = block.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def ramp(start, end, steps):
    def schedule(step):
        if step >= steps - 1:
            return end
        frac = step / (steps - 1)
        return start + (end - start) * frac
    return schedule


def project(activations, direction):
    unit = direction / direction.norm()
    return activations @ unit
