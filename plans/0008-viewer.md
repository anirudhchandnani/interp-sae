# 0008 Viewer

## Problem

The toolkit can assemble its output into a single JSON report, but reading that
report still means opening the raw file by hand. A report holds three different
shapes at once: per-feature statistics, the top activating tokens for each of the
top features, and short steering continuations across a few coefficients. We want
a small viewer that loads a saved report and lets a person browse those three
parts without writing any code, and we want the shaping that feeds it to be plain,
testable functions rather than logic buried in UI callbacks.

## Approach

The work splits in two. A pure module, `interp_sae.display`, turns a loaded report
dict into display-ready rows, and a thin Streamlit app renders them.

`feature_rows` reads the `stats` list and returns one row per feature with its id,
frequency, and mean activation, keeping the frequency-descending order the report
was built with. `token_rows` looks a feature id up in the `top_tokens` section,
keyed by the string form of the id, and returns its records as rows of token text,
activation, prompt index, and position, already in descending activation order.
`steering_rows` returns one row per steering example with its coefficient and text.
`feature_ids` returns the integer feature ids present in the token section, which
the app uses to populate a selector. Every helper is defensive: a missing or empty
section returns an empty list rather than raising, and none of them import
Streamlit.

`app.py` is a uv inline-dependency script that depends only on Streamlit. It takes
a report through a file uploader or a path text input, parses the upload or calls
`load_report`, and shows a header, a table of feature rows, a selectbox of feature
ids, a table of token rows for the selected feature, and a table of steering rows.
When nothing is loaded yet it shows a short instruction instead of crashing, and
it does no model work at all; it only reads a report file.

## Result

The display helpers are exported from the package and covered by offline checks
that build a report in the exact schema `build_report` produces and assert that
feature rows come back one per feature in order with the right fields, that token
rows for a known feature are in descending activation order with the right fields,
that steering rows carry one coefficient and text each, that the feature ids match,
and that each helper returns an empty list when its section is absent. Run the
viewer with:

```
uv run streamlit run app.py
```
