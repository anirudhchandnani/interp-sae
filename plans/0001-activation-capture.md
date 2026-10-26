# 0001 Activation capture

## Problem

The toolkit will train a sparse autoencoder, fit linear probes, and steer a model
at inference time. All three need the model's internal activations as input. Before
any of that we need a reliable way to pull hidden activations out of a small
GPT2-style transformer for a batch of prompts.

## Approach

Attach forward hooks to each transformer block and capture two points per layer:
the residual stream output of the block and the output of the block's MLP. Hooks
read tensors during a single forward pass and detach them so nothing holds the
graph. Results come back stacked per layer so later stages can index by layer and
point. Hooks are always removed after the pass, even on error.

## Result

`capture_activations` takes a model and tokenized inputs and returns a dict keyed
by point (`resid`, `mlp`), each a tensor of shape (layers, batch, sequence, hidden).
A text helper tokenizes prompts first. Tests build a tiny random GPT2 in memory and
check the shapes, the layer count, and that hooks are cleaned up, all offline.
