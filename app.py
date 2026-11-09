# /// script
# requires-python = ">=3.10"
# dependencies = ["streamlit>=1.37"]
# ///
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interp_sae.display import feature_rows, token_rows, steering_rows, feature_ids
from interp_sae.report import load_report


def read_report(uploaded, path):
    if uploaded is not None:
        return json.loads(uploaded.getvalue().decode("utf-8"))
    if path:
        return load_report(path)
    return None


st.title("interp-sae report viewer")

uploaded = st.file_uploader("Upload a report", type="json")
path = st.text_input("Or enter a report path")

report = None
try:
    report = read_report(uploaded, path.strip())
except (OSError, ValueError) as error:
    st.error(f"Could not load report: {error}")

if report is None:
    st.info("Upload a report file or enter a path to a saved report to begin.")
else:
    st.header("Features")
    st.table(feature_rows(report))

    ids = feature_ids(report)
    if ids:
        selected = st.selectbox("Feature", ids)
        st.header("Top activating tokens")
        st.table(token_rows(report, selected))

    rows = steering_rows(report)
    if rows:
        st.header("Steering examples")
        st.table(rows)
