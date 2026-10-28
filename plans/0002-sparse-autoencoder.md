# 0002 Sparse autoencoder

## Problem

Captured activations are dense and hard to read directly. We want a sparse
dictionary over them so individual units stand for something interpretable. That
means a model that reconstructs an activation from a small number of active
features, which later stages can train on real captures and inspect.

## Approach

A tied-bias relu sparse autoencoder. Encoding subtracts a decoder bias, applies a
linear map, and passes the result through a relu, so feature activations are
non-negative. Decoding maps features back and adds the same bias. An l1 penalty on
the feature activations pushes most of them toward zero, leaving a sparse code. A
loss method returns the reconstruction error, the l1 term, and their weighted sum
so a future training loop can optimize it.

## Result

`SparseAutoencoder` is an `nn.Module` taking an input dimension and a hidden
feature dimension. `forward` returns the reconstruction and the feature
activations, and `loss` returns a dict of recon, l1, and total. Tests build a small
model on random input, check the feature and reconstruction shapes, confirm
features are non-negative and losses are finite, and run a handful of gradient
steps to confirm the reconstruction loss drops. No model and no network, so it runs
offline and fast.
