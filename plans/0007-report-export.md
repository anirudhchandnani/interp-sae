# 0007 Browsing and export layer

## Problem

The toolkit can capture activations, fit a sparse autoencoder, probe and steer a
model, and calibrate a dose, but the results only live as tensors inside a Python
session. There is nothing a person can open and read, and nothing for the viewer
in the next step to consume. We want a small layer that turns the toolkit output
into inspectable, JSON-serializable artifacts: which tokens light up a feature,
how often each feature fires, and what steering a prompt does at a few
coefficients. This is the data layer only, not the viewer.

## Approach

Four read-only helpers plus an assemble-and-persist pair, all reusing the existing
modules and emitting plain Python values.

`top_activating_tokens` captures activations for a list of prompts at one point
and layer, encodes them with the autoencoder to get per-token feature activations,
and for each requested feature id walks every prompt and position to collect the
top-k occurrences. Each occurrence is a record with the feature id, the activation
value, the prompt index, the token position, and the decoded token text. Records
are sorted by activation descending and truncated to k. The return is a dict keyed
by feature id to its list of records.

`feature_stats` takes an (N, hidden) matrix, encodes it once, and reports per
feature the activation frequency (fraction of rows where the feature is above
zero) and the mean activation, as a list of dicts sorted by frequency descending.

`steering_examples` greedily generates a short continuation for a prompt under
`steer` at each coefficient and returns a list of {coef, text} dicts. Generation
is kept short so the pass stays fast.

`build_report` ties these together: it computes feature_stats over the data
matrix (capturing and flattening it from the prompts when one is not supplied),
takes the top features by frequency, gathers their top activating tokens, and
optionally runs steering_examples. `save_report` and `load_report` write and read
the result with `json.dump` and `json.load`.

Every value returned is JSON-serializable: tensors are reduced to Python floats
and ints with `.item()` and `.tolist()`, and the report dict keys the top tokens
by the string form of the feature id so a round trip through JSON compares equal.

## Export schema

The report is a single JSON object:

- `point`: string, the capture point ("resid" or "mlp").
- `layer`: int, the block index the activations came from.
- `stats`: list of `{feature: int, frequency: float, mean: float}`, sorted by
  frequency descending, one entry per autoencoder feature.
- `top_tokens`: object keyed by feature id as a string, each value a list of
  `{feature: int, activation: float, prompt: int, position: int, token: str}`
  records sorted by activation descending, at most k per feature.
- `steering` (optional): list of `{coef: float, text: str}`, one per coefficient.

## Result

`top_activating_tokens`, `feature_stats`, `steering_examples`, `build_report`,
`save_report`, and `load_report` are exported, so the toolkit output becomes a
single JSON file a reviewer can read and the viewer can load. Tests build a tiny
in-memory GPT2-style model with a stub tokenizer and synthetic activations so they
run offline and fast. They check that the top-token records carry the expected
keys and JSON-serializable types and sort by activation descending, that the top
record points at the true maximum computed independently from the same model and
autoencoder, that feature statistics match hand-set frequencies and stay within
range, that steering returns one entry per coefficient with the matching
coefficient and a string, and that a built report survives a save and load round
trip and is valid JSON.
