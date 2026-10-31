# 0005 Dose calibration and dose ramps

## Problem

Steering adds `coef * direction` into the residual stream, but a single
coefficient is a blunt instrument. Too little and the concept barely shifts the
output; too much and the model collapses into repetition or nonsense. We want to
pick the strongest coefficient that still keeps generation coherent, and we want
the coefficient to be able to change as generation proceeds rather than staying
fixed for the whole run.

## Approach

Two pieces. First, dose ramps let the steering coefficient vary by generation
step. The `steer` hook keeps an internal forward-pass counter scoped to the
context, so on the nth forward pass through the steered block it applies
`coef(n)` when `coef` is callable and the plain constant otherwise. The counter
starts at zero on every `steer` entry and is dropped when the context exits, so
nothing leaks between runs. `ramp(start, end, steps)` returns a schedule that
interpolates linearly from `start` at step 0 to `end` at step `steps - 1` and
clamps to `end` past that range. A plain float still works unchanged.

Second, a calibration harness sweeps coefficients and scores each with two
model-internal proxies, so there is no external judge and no second large model.
`strength` runs a short greedy generation under steering, captures the residual
at the chosen layer, and returns the mean projection onto the unit direction;
higher means the concept is more present, and because steering adds
`coef * direction` the projection grows with the coefficient. `coherence`
combines the mean token log-probability the model assigns to the sequence
(higher is better) with a penalty equal to the fraction of repeated word
trigrams in the decoded text (subtracted), so a fluent sequence scores above a
degenerate repetitive one. These two pull against each other: pushing the
coefficient up raises strength but eventually drives coherence down, and the job
of calibration is to find where that tradeoff turns.

`calibrate` generates under each coefficient, records strength and coherence in a
table, and selects the dose as the largest coefficient whose coherence stays at
or above the threshold. The default threshold is anchored on the unsteered
`coef = 0` baseline, allowing a small relative drop below it; passing
`min_coherence` overrides this with an absolute floor. If no steered coefficient
clears the threshold it falls back to the smallest coefficient, so the result is
never an incoherent dose.

## Result

`steer` now accepts a per-step schedule as well as a constant, `ramp` builds a
linear schedule, and `calibrate` returns a per-coefficient table together with
the selected dose, reusing `steer`, `project`, `capture_activations`, and the
existing capture path rather than reimplementing hooks. Tests build a tiny
in-memory GPT2-style model so they run offline and fast. Because a random tiny
model has no real semantics, they check mechanics and monotonic properties: the
scheduled coefficient applied at the first versus a later forward pass, that a
constant float is unchanged, that the counter resets between contexts and the
hook is removed, that strength grows with the coefficient, that heavy trigram
repetition lowers coherence, and that calibrate returns a full table and a
selected dose from the provided coefficients that is not forced to the most
extreme one.
