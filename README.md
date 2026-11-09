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
- Activation steering: add a scaled direction into the residual stream at a chosen
  block during a forward pass, and project activations onto a direction to monitor
  its presence.
- Dose calibration and dose ramps: sweep steering coefficients and pick the
  strongest one that keeps generation coherent, and vary the coefficient across
  generation steps with a per-step schedule.
- Sparse autoencoder training: flatten captured activations into per-token rows
  and fit the autoencoder with a minibatch Adam loop over its total loss.
- Browsing and export layer: gather the top activating tokens for a feature,
  per-feature activation frequency and mean, and short steering continuations, and
  assemble them into a single JSON report that saves and loads for later browsing.
- Report viewer: a small Streamlit app that loads a saved report and browses its
  feature statistics, top activating tokens per feature, and steering examples.

Run the viewer:

```
uv run streamlit run app.py
```

## Possible next steps

- Compare two reports side by side to see how features shift between runs.
