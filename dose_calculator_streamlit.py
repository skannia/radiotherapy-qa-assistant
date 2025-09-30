# dose_calculator_streamlit.py
# MU Calculator with web UI using Streamlit

import streamlit as st

# Reference data (simplified)
percent_depth_dose = {
    1: 100.0,
    5: 87.0,
    10: 67.0,
    15: 54.0,
    20: 44.0
}

sc_factors = {
    "10x10": 1.00,
    "20x20": 1.05,
    "5x5": 0.95
}

def calculate_mu(prescribed_dose, dose_rate, depth_cm, field_size="10x10"):
    if depth_cm in percent_depth_dose:
        pdd = percent_depth_dose[depth_cm] / 100.0
    else:
        closest = min(percent_depth_dose.keys(), key=lambda x: abs(x - depth_cm))
        pdd = percent_depth_dose[closest] / 100.0

    sc = sc_factors.get(field_size, 1.0)
    mu = prescribed_dose / (dose_rate * pdd * sc)
    return mu

def compare_with_tps(calculated_mu, tps_mu, tolerance=0.03):
    deviation = (calculated_mu - tps_mu) / tps_mu
    within_tol = abs(deviation) <= tolerance
    return deviation * 100, within_tol

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="MU Calculator", page_icon="⚡", layout="centered")

st.title("⚡ MU Calculator for Medical Physics QA")
st.write("A simple QA tool to compare manual MU calculation with TPS output.")

# Input fields
prescribed_dose = st.number_input("Prescribed Dose (cGy):", min_value=0.0, value=200.0)
dose_rate = st.number_input("Dose Rate (cGy/MU):", min_value=0.0, value=1.0)
depth_cm = st.number_input("Depth (cm):", min_value=0.0, value=10.0)
field_size = st.selectbox("Field Size:", list(sc_factors.keys()))
tps_mu = st.number_input("TPS MU:", min_value=0.0, value=300.0)

if st.button("Calculate"):
    calc_mu = calculate_mu(prescribed_dose, dose_rate, depth_cm, field_size)
    deviation, within_tol = compare_with_tps(calc_mu, tps_mu)

    st.success(f"Calculated MU: **{calc_mu:.2f}**")
    st.info(f"TPS MU: **{tps_mu:.2f}**")
    st.write(f"Deviation: **{deviation:.2f}%**")
    if within_tol:
        st.markdown("✅ **Within tolerance**")
    else:
        st.markdown("⚠️ **Out of tolerance!**")
