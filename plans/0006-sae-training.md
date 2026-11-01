# 0006 Sparse autoencoder training on captured activations

## Problem

The sparse autoencoder already knows how to encode an activation into a sparse
feature code, reconstruct it, and report recon, l1, and total losses, but nothing
actually fits it. A smoke test shows the recon loss can fall under a plain
optimizer, yet there is no reusable training loop and no path from captured model
activations to the matrix the autoencoder needs. We want to take the activations
that capture already produces and fit the autoencoder on them.

## Approach

Three small pieces. First, capture returns per-layer tensors of shape
(n_layer, batch, seq, hidden) keyed by point. `flatten_activations` selects one
point and layer and reshapes the (batch, seq, hidden) slice into (batch*seq,
hidden), so every token becomes an independent training row and the row order
follows batch then sequence position.

Second, `train_sae` runs an Adam loop over the (N, hidden) matrix. Each step
samples a random minibatch of row indices, computes the autoencoder total loss on
that batch, which is reconstruction error plus the l1 coefficient times the mean
feature magnitude, and takes an optimizer step. It records the per-step recon,
l1, and total as detached floats and returns that history. The l1 term pushes the
feature code toward zero, so a positive coefficient trades a little
reconstruction for a much sparser code, and with the coefficient at zero the code
stays dense.

Third, `train_on_model` ties it together: it captures activations for a list of
prompts, flattens them at the chosen point and layer, sizes a new autoencoder
from the hidden width and a hidden ratio, trains it, and returns the fitted model
with its history. It reuses the capture path, `flatten_activations`, and the
existing autoencoder rather than reimplementing any of them.

## Result

`flatten_activations`, `train_sae`, and `train_on_model` are exported, so a few
prompts go in and a fitted sparse autoencoder with a loss history comes out.
Tests build a tiny in-memory GPT2-style model and low-rank synthetic vectors so
they run offline and fast. They check that flattening maps tokens to rows with
the right values, that training drives the total loss below its starting point
and returns a finite history of the expected length, that a positive l1
coefficient yields a sparser code than a zero coefficient, and that the
end-to-end path returns an autoencoder whose input width matches the model hidden
size.
