"""Person 5 Streamlit stub — Fraud Analyst Workbench."""

import streamlit as st

st.set_page_config(page_title="Fraud Analyst Workbench", layout="wide")
st.title("Fraud Analyst Workbench")
st.info("Person 5: load models/metrics.json + scored queue after Person 3 trains.")
st.markdown(
    """
- High / Medium / Low risk counts
- Ranked alert queue
- Reason codes
- Customer behavioural timeline (Plotly)
- PR curve from `models/pr_curve.png`
"""
)
