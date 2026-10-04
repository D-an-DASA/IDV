import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Filters")

st.sidebar.selectbox(
    "Gender",
    ["All", "Male", "Female"]
)

st.sidebar.slider(
    "Age Range",
    18,
    100,
    (18, 100)
)

st.sidebar.selectbox(
    "Cardiovascular Disease",
    ["All", "Yes", "No"]
)


# ============================================================
# HEADER
# ============================================================

st.title("Cardiovascular Risk Dashboard")

st.write(
    "Explore demographic, medical, and lifestyle factors "
    "associated with cardiovascular disease."
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Patients", "68,677")

with col2:
    st.metric("Cardiovascular Risk", "—")

with col3:
    st.metric("Average Age", "—")

with col4:
    st.metric("Average BMI", "—")


# ============================================================
# SECTION 1 — DEMOGRAPHICS
# ============================================================

st.subheader("Demographic Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Age Distribution")
    st.empty()

with col2:
    st.write("### Gender Distribution")
    st.empty()


# ============================================================
# SECTION 2 — MEDICAL FACTORS
# ============================================================

st.subheader("Medical Risk Factors")

col1, col2 = st.columns(2)

with col1:
    st.write("### Blood Pressure")
    st.empty()

with col2:
    st.write("### Cholesterol / Glucose")
    st.empty()


# ============================================================
# SECTION 3 — LIFESTYLE FACTORS
# ============================================================

st.subheader("Lifestyle Factors")

col1, col2 = st.columns(2)

with col1:
    st.write("### Physical Activity")
    st.empty()

with col2:
    st.write("### Smoking / Alcohol")
    st.empty()


# ============================================================
# SECTION 4 — RISK ANALYSIS
# ============================================================

st.subheader("Risk Factor Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Risk Factor Relationship")
    st.empty()

with col2:
    st.write("### Correlation Analysis")
    st.empty()


# ============================================================
# SECTION 5 — GEOGRAPHIC ANALYSIS
# ============================================================

st.subheader("Geographic Analysis")

st.write("### Cardiovascular Risk by Location")

st.empty()


# ============================================================
# SECTION 6 — PREDICTION
# ============================================================

st.subheader("Cardiovascular Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    st.write("### Predicted Risk")
    st.empty()

with col2:
    st.write("### Prediction Trend")
    st.empty()


# ============================================================
# DATA TABLE
# ============================================================

st.subheader("Patient Data")

st.empty()
