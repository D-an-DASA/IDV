import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- Header ---------- */

    .eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .page-description {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* ---------- Cards ---------- */

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 180px;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.2rem;
    }

    .card-subtitle {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    /* ---------- KPI ---------- */

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.1rem;
        min-height: 125px;
    }

    .kpi-label {
        color: #6b7280;
        font-size: 0.82rem;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .kpi-detail {
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }

    /* ---------- Section headings ---------- */

    .section-heading {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.85rem;
    }

    /* ---------- Prediction ---------- */

    .prediction-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 360px;
    }

    .prediction-title {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .prediction-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 240px;
        color: #9ca3af;
        text-align: center;
    }

    /* ---------- Footer ---------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
        color: #9ca3af;
        font-size: 0.75rem;
        display: flex;
        justify-content: space-between;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — GLOBAL FILTERS
# ============================================================

with st.sidebar:

    st.markdown("## ❤️ CardioView")
    st.caption("Population health")

    st.divider()

    st.markdown("### ⚙️ Global filters")

    sex = st.selectbox(
        "Sex",
        ["All sexes", "Female", "Male"],
    )

    age_group = st.selectbox(
        "Age group",
        [
            "All age groups",
            "18–24",
            "25–34",
            "35–44",
            "45–54",
            "55–64",
            "65–74",
            "75+",
        ],
    )

    high_bp = st.selectbox(
        "High BP",
        ["Any status", "No", "Yes"],
    )

    high_chol = st.selectbox(
        "High cholesterol",
        ["Any status", "No", "Yes"],
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["Any status", "No", "Prediabetes", "Yes"],
    )

    smoker = st.selectbox(
        "Smoking status",
        ["Any status", "Non-smoker", "Smoker"],
    )

    physical_activity = st.selectbox(
        "Physical activity",
        ["Any activity", "Active", "Inactive"],
    )

    income = st.selectbox(
        "Income",
        [
            "All income levels",
            "< $25k",
            "$25–50k",
            "$50–75k",
            "$75k+",
        ],
    )

    education = st.selectbox(
        "Education",
        [
            "All education levels",
            "High school",
            "Some college",
            "College graduate",
        ],
    )

    bmi_range = st.slider(
        "BMI range",
        min_value=17.0,
        max_value=40.0,
        value=(17.0, 40.0),
        step=0.5,
    )

    st.divider()

    apply_filters = st.button(
        "✓ Apply filters",
        use_container_width=True,
        type="primary",
    )

    reset_filters = st.button(
        "↻ Reset all",
        use_container_width=True,
    )

    st.divider()

    st.caption("DATA COVERAGE")
    st.progress(0.0)
    st.caption("Dataset information will be added later.")


# ============================================================
# HEADER
# ============================================================

header_col, action_col = st.columns([4, 1])

with header_col:

    st.markdown(
        '<div class="eyebrow">POPULATION HEALTH OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Cardiovascular Risk Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-description">'
        'Explore demographic, health, lifestyle, socioeconomic, '
        'and predicted risk factors.'
        '</div>',
        unsafe_allow_html=True,
    )

with action_col:

    st.caption("LAST UPDATED")
    st.write("—")
    st.button("↓ Export", use_container_width=True)


# ============================================================
# KPI CARDS
# ============================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">—</div>
            <div class="kpi-detail">Waiting for dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">CVD Rate</div>
            <div class="kpi-value">—</div>
            <div class="kpi-detail">HeartDiseaseorAttack</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Largest Age Group</div>
            <div class="kpi-value">—</div>
            <div class="kpi-detail">Waiting for dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Average BMI</div>
            <div class="kpi-value">—</div>
            <div class="kpi-detail">Waiting for dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FEATURE AREA
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <div class="section-title">Risk overview & prediction</div>
        <div class="section-description">
            Population-level cardiovascular risk overview and model prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col, prediction_col = st.columns(2)


# ---------- Risk Overview ----------

with overview_col:

    st.markdown(
        """
        <div class="card">
            <div class="eyebrow">POPULATION RISK OVERVIEW</div>
            <div class="card-title">Cardiovascular Risk Overview</div>
            <div class="card-subtitle">
                Population risk visualization will be added here.
            </div>

            <div class="prediction-placeholder">
                Risk overview placeholder
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Prediction ----------

with prediction_col:

    st.markdown(
        """
        <div class="prediction-card">
            <div class="eyebrow">XGBOOST RISK PREDICTION</div>
            <div class="prediction-title">
                Cardiovascular risk model
            </div>

            <div class="prediction-placeholder">
                Prediction interface will be added here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <div class="eyebrow">POPULATION INSIGHTS</div>
        <div class="section-title">Cardiovascular risk factors</div>
        <div class="section-description">
            Demographics → health → lifestyle → socioeconomic factors → interactions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHART ROW 1
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Risk by Age Group</div>
            <div class="card-subtitle">
                CVD prevalence across age categories
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


with col2:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Risk by Sex</div>
            <div class="card-subtitle">
                CVD prevalence between sexes
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CHART ROW 2
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                High BP vs Cardiovascular Disease
            </div>
            <div class="card-subtitle">
                CVD prevalence by High BP status
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


with col2:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                BMI and Cardiovascular Risk
            </div>
            <div class="card-subtitle">
                CVD prevalence by BMI category
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CHART ROW 3
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Health Condition Burden</div>
            <div class="card-subtitle">
                CVD prevalence with key health conditions
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


with col2:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Lifestyle Risk Factors</div>
            <div class="card-subtitle">
                Smoking, activity, diet, and alcohol
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CHART ROW 4
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Socioeconomic & Healthcare
            </div>
            <div class="card-subtitle">
                CVD prevalence across access and socioeconomic factors
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


with col2:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Risk Factor Interactions</div>
            <div class="card-subtitle">
                High BP × High cholesterol → CVD
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FILTERED POPULATION SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <div class="eyebrow">AGGREGATED DATA</div>
        <div class="section-title">Filtered Population Summary</div>
        <div class="section-description">
            Key statistics for the current filtered population.
            No individual records are shown.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)

summary_items = [
    ("Records", "—", "Current population"),
    ("CVD rate", "—", "HeartDiseaseorAttack"),
    ("Average BMI", "—", "Filtered population"),
    ("High BP", "—", "Population prevalence"),
]

for col, (label, value, detail) in zip(summary_cols, summary_items):

    with col:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <span>
            CardioView Analytics · Data is de-identified and intended
            for analytical use only.
        </span>

        <span>
            Model — 
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)