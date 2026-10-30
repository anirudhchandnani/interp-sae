# 0004 Activation steering

## Problem

Capture and probes tell us which directions in activation space carry a concept,
but reading is only half the loop. To check that a direction is causal rather than
merely correlated, we want to add it back into the model mid-forward and watch the
behavior shift. We also want a cheap readout of how strongly a direction is present
in an activation so we can watch that shift quantitatively.

## Approach

A forward hook on a chosen block in `model.transformer.h` adds `coef * direction`
to that block's residual output. The block output is a tuple, so the hook rewrites
`output[0]` and passes the rest through unchanged. The direction is a 1D tensor of
size `n_embd` and broadcasts across batch and sequence, so every token at that
layer gets the same nudge. The hook is registered inside a context manager and
removed in a `finally` block, so a steered forward never leaks a hook. To monitor
presence, `project` normalizes the direction and takes the dot product with the
activations, giving a per-token scalar that measures alignment with the direction.

## Result

`steer(model, layer, direction, coef)` is a context manager that installs the hook
on entry and removes it on exit; any forward pass or generation run inside the block
is steered. `project(activations, direction)` returns the per-token projection onto
the unit direction. Tests build a tiny in-memory GPT2-style model so they run
offline and fast: they confirm that a nonzero coefficient changes the output logits
while a zero coefficient leaves them unchanged, that the hook is gone after the
context exits, that the captured residual at the steered layer moves by exactly
`coef * direction`, and that projecting a vector onto its own direction recovers its
norm. Dose calibration and step-dependent dose ramps are planned follow-ups and are
not implemented here.
