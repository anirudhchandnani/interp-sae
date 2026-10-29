# interp-sae

A small interpretability toolkit built around a local GPT2-style transformer. The
goal is a compact, readable set of tools for looking inside a model and acting on
what you find: training a sparse autoencoder on its activations, fitting linear
probes to read off concepts, steering the model by adding directions back into its
activations, and a viewer to browse the results.

## Quickstart

Everything runs through [uv](https://docs.astral.sh/uv/) with inline dependencies,
so there is no separate environment to manage.

Run the tests:

```
uv run tests/test_capture.py
```

Capture activations from a few prompts:

```python
from interp_sae import load_model, capture_from_text

model, tokenizer = load_model("distilgpt2")
acts = capture_from_text(model, tokenizer, ["the cat sat", "hello world"])
print(acts["resid"].shape)  # (layers, batch, sequence, hidden)
```

## What works now

- Activation capture: pull residual-stream and MLP activations from each block of a
  small GPT2-style model using forward hooks, for a batch of prompts, returned as
  per-layer tensors.
- Sparse autoencoder: a tied-bias relu model that encodes an activation into
  non-negative feature activations and reconstructs it, with recon, l1, and total
  losses ready for a training loop.
- Linear probes: fit a logistic regression on activations to read off a binary
  concept and recover its direction in activation space, with accuracy and
  probability readouts.

## Roadmap

- Sparse autoencoder training on captured activations
- Activation steering at inference time
- A viewer to browse features, probes, and steering effects
