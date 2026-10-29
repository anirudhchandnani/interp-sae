# 0003 Linear probe

## Problem

Captured activations hold concepts in a distributed form, but we often want a
single yes or no readout: does this activation carry a given binary concept. We
also want the answer as a direction in activation space, so it composes with
capture and later steering rather than living in a separate feature basis.

## Approach

A logistic regression probe: one linear layer mapping an activation vector to a
single logit, trained with binary cross entropy on gradient descent. Fitting the
probe recovers a separating hyperplane whose normal is the concept direction. The
normalized weight vector is exposed directly so it can be added back into
activations downstream. Prediction passes logits through a sigmoid for
probabilities and thresholds at one half for labels.

## Result

`LinearProbe` is an `nn.Module` taking an input dimension. `fit` runs the training
loop, `predict_proba` and `predict` return probabilities and labels, `accuracy`
reports the fraction correct, and `direction` returns the normalized weight vector.
Tests build two linearly separable Gaussian clusters in a small dimension, fit the
probe, and confirm the accuracy is high, the recovered direction aligns with the
true separating direction, the outputs have the right shapes with probabilities in
range, and the fit improves accuracy over initialization. Synthetic data means no
model and no network, so it runs offline and fast.
