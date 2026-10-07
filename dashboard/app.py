import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "processed" / "heart_disease_health_indicators_BRFSS2015.csv"
MODEL = BASE / "src" / "model" / "XGBoost" / "model_detail" / "xgboost_model.pkl"
META = BASE / "src" / "model" / "XGBoost" / "model_detail" / "model_metadata.json"
TARGET = "HeartDiseaseorAttack"

st.set_page_config(page_title="Cardiovascular Risk Dashboard", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    "<style>.block-container{max-width:1500px;padding-top:4rem}"
    ".eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.08em;"
    "text-transform:uppercase;color:#9ca3af}"
    ".page-title{font-size:2rem;font-weight:700}"
    ".sub{color:#9ca3af;font-size:.85rem;margin-bottom:.8rem}"
    "[data-testid=stMetric]{border:1px solid #374151;border-radius:12px;"
    "padding:.8rem 1rem}</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- labels
AGE_ORDER = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
             "55–59", "60–64", "65–69", "70–74", "75–79", "80+"]
AGE_LBL = dict(enumerate(AGE_ORDER, 1))
AGE_BINS = {"18–24": [1], "25–34": [2, 3], "35–44": [4, 5], "45–54": [6, 7],
            "55–64": [8, 9], "65–74": [10, 11], "75+": [12, 13]}
INC_LBL = {1: "<$10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k", 5: "$25–35k",
           6: "$35–50k", 7: "$50–75k", 8: "$75k+"}
INC_BINS = {"< $25k": [1, 2, 3, 4], "$25–50k": [5, 6], "$50–75k": [7], "$75k+": [8]}
EDU_LBL = {1: "No school", 2: "Elementary", 3: "Some high school", 4: "High school grad",
           5: "Some college", 6: "College grad"}
EDU_BINS = {"High school or less": [1, 2, 3, 4], "Some college": [5],
            "College graduate": [6]}
DIA_LBL = {0: "No", 1: "Prediabetes", 2: "Yes"}
GEN_LBL = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
SEX_ORDER = ["Female", "Male"]


@st.cache_data
def load():
    d = pd.read_csv(CSV)
    d = d[d.BMI.isna() | d.BMI.between(12, 60)].reset_index(drop=True)
    d["Age_lbl"] = pd.Categorical(d.Age.map(AGE_LBL), AGE_ORDER, ordered=True)
    d["Sex_lbl"] = d.Sex.map({0: "Female", 1: "Male"})
    d["Inc_lbl"] = pd.Categorical(d.Income.map(INC_LBL), list(INC_LBL.values()), ordered=True)
    d["Edu_lbl"] = pd.Categorical(d.Education.map(EDU_LBL), list(EDU_LBL.values()), ordered=True)
    d["CVD"] = d[TARGET].map({0: "No CVD", 1: "CVD"})
    return d


@st.cache_resource
def get_model():
    return joblib.load(MODEL), json.load(open(META, encoding="utf-8"))


@st.cache_data
def predict_all():
    model, meta = get_model()
    return model.predict_proba(load()[meta["features"]])[:, 1]


data = load()
model, meta = get_model()
proba = pd.Series(predict_all(), index=data.index)
THR = meta["threshold"]

# ---------------------------------------------------------------- sidebar
DEFAULTS = {"f_sex": "All sexes", "f_age": "All age groups", "f_bp": "Any",
            "f_chol": "Any", "f_dia": "Any", "f_smk": "Any", "f_act": "Any",
            "f_inc": "All income levels", "f_edu": "All education levels",
            "f_bmi": (12.0, 60.0), "ver": 0}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    ver = st.session_state.ver + 1
    st.session_state.update(DEFAULTS)
    st.session_state.ver = ver


def clear_selection():
    st.session_state.ver += 1


with st.sidebar:
    st.markdown("## CardioView")
    st.caption("Population health")
    st.divider()
    st.markdown("### ⚙️ Global filters")
    st.selectbox("Sex", ["All sexes", "Female", "Male"], key="f_sex")
    st.selectbox("Age group", ["All age groups"] + list(AGE_BINS), key="f_age")
    st.selectbox("High BP", ["Any", "No", "Yes"], key="f_bp")
    st.selectbox("High cholesterol", ["Any", "No", "Yes"], key="f_chol")
    st.selectbox("Diabetes", ["Any", "No", "Prediabetes", "Yes"], key="f_dia")
    st.selectbox("Smoker", ["Any", "No", "Yes"], key="f_smk")
    st.selectbox("Physically active", ["Any", "No", "Yes"], key="f_act")
    st.selectbox("Income", ["All income levels"] + list(INC_BINS), key="f_inc")
    st.selectbox("Education", ["All education levels"] + list(EDU_BINS), key="f_edu")
    st.slider("BMI range", 12.0, 60.0, step=0.5, key="f_bmi")
    st.divider()
    st.button("↻ Reset all", width="stretch", on_click=reset)
    st.caption("DATA COVERAGE")


def yn(d, col, v):
    return d if v == "Any" else d[d[col] == (1 if v == "Yes" else 0)]


S = st.session_state
base = data
if S.f_sex != "All sexes":
    base = base[base.Sex_lbl == S.f_sex]
if S.f_age != "All age groups":
    base = base[base.Age.isin(AGE_BINS[S.f_age])]
if S.f_inc != "All income levels":
    base = base[base.Income.isin(INC_BINS[S.f_inc])]
if S.f_edu != "All education levels":
    base = base[base.Education.isin(EDU_BINS[S.f_edu])]
if S.f_dia != "Any":
    base = base[base.Diabetes == {"No": 0, "Prediabetes": 1, "Yes": 2}[S.f_dia]]
base = yn(yn(yn(yn(base, "HighBP", S.f_bp), "HighChol", S.f_chol), "Smoker", S.f_smk),
          "PhysActivity", S.f_act)
base = base[base.BMI.isna() | base.BMI.between(*S.f_bmi)]
with st.sidebar:
    st.progress(len(base) / len(data))
    st.caption(f"{len(base):,} of {len(data):,} records match the filters.")


# ---------------------------------------------------------------- cross-filter
def picked(key, order):
    ev = st.session_state.get(key)
    if not ev:
        return []
    return [order[p["point_number"]] for p in ev["selection"]["points"]
            if p["point_number"] < len(order)]


AGE_KEY, SEX_KEY = f"age_{S.ver}", f"sex_{S.ver}"
age_sel, sex_sel = picked(AGE_KEY, AGE_ORDER), picked(SEX_KEY, SEX_ORDER)


def by_age(d):
    return d[d.Age_lbl.isin(age_sel)] if age_sel else d


def by_sex(d):
    return d[d.Sex_lbl.isin(sex_sel)] if sex_sel else d


df = by_sex(by_age(base))


def sty(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def rate_yes_no(d, cols):
    rows = []
    for lbl, s in cols.items():
        for v, name in [(1, "Yes"), (0, "No")]:
            m = d[TARGET][s.reindex(d.index) == v]
            rows.append((lbl, name, m.mean() * 100 if len(m) else np.nan))
    return pd.DataFrame(rows, columns=["Factor", "Answer", "CVD rate (%)"])


def yn_bar(d, cols, title):
    fig = px.bar(rate_yes_no(d, cols), x="Factor", y="CVD rate (%)", color="Answer",
                 barmode="group", title=title,
                 color_discrete_map={"Yes": "#ef4444", "No": "#38bdf8"})
    fig.update_layout(legend_title_text="Has factor?")
    return sty(fig)


# ---------------------------------------------------------------- header
st.markdown('<div class="eyebrow">POPULATION HEALTH OVERVIEW</div>'
            '<div class="page-title">Cardiovascular Risk Dashboard</div>'
            '<div class="sub">Explore demographic, health, lifestyle, socioeconomic '
            'and predicted risk factors. Click a bar or slice to cross-filter.</div>',
            unsafe_allow_html=True)

if len(df) == 0:
    st.warning("No records match the current filters.")
    st.stop()

if age_sel or sex_sel:
    c1, c2 = st.columns([5, 1])
    c1.info("Cross-filter active: " + " · ".join(age_sel + sex_sel))
    c2.button("Clear selection", on_click=clear_selection, width="stretch")

k = st.columns(4)
k[0].metric("Total Records", f"{len(df):,}")
k[1].metric("CVD Rate", f"{df[TARGET].mean() * 100:.1f}%")
k[2].metric("Largest Age Group", str(df.Age_lbl.value_counts().idxmax()))
k[3].metric("Average BMI", f"{df.BMI.mean():.1f}")

# ---------------------------------------------------------------- risk + prediction
st.markdown("### Risk overview & prediction")
left, right = st.columns(2)

with left.container(border=True):
    st.markdown("**Population Risk Overview**")
    st.caption(f"Model-predicted risk for the filtered population "
               f"(threshold {THR:.0%}).")
    p = proba.loc[df.index]
    fig = px.histogram(p * 100, nbins=50, labels={"value": "Predicted risk (%)"})
    fig.add_vline(x=THR * 100, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Threshold")
    fig.update_layout(showlegend=False, yaxis_title="People")
    st.plotly_chart(sty(fig, 260), width="stretch")
    st.metric("Flagged high-risk", f"{(p >= THR).mean() * 100:.1f}%")

with right.container(border=True):
    st.markdown("**XGBoost risk prediction**")
    with st.form("predict"):
        a, b, c = st.columns(3)
        sex = a.selectbox("Sex", SEX_ORDER)
        age = b.selectbox("Age", AGE_ORDER, index=6)
        bmi = c.number_input("BMI", 12.0, 60.0, 27.0)
        edu = a.selectbox("Education", list(EDU_LBL.values()), index=5)
        inc = b.selectbox("Income", list(INC_LBL.values()), index=6)
        dia = c.selectbox("Diabetes", list(DIA_LBL.values()))
        gen = a.select_slider("General health", list(GEN_LBL.values()), "Good")
        ment = b.slider("Bad mental days", 0, 30, 0)
        phys = c.slider("Bad physical days", 0, 30, 0)
        t = st.columns(4)
        flags = {
            "HighBP": t[0].toggle("High BP"), "HighChol": t[1].toggle("High chol."),
            "CholCheck": t[2].toggle("Chol. check", True), "Smoker": t[3].toggle("Smoker"),
            "Stroke": t[0].toggle("Stroke"), "DiffWalk": t[1].toggle("Hard to walk"),
            "PhysActivity": t[2].toggle("Active", True), "Fruits": t[3].toggle("Fruits"),
            "Veggies": t[0].toggle("Veggies", True), "HvyAlcoholConsump": t[1].toggle("Heavy alcohol"),
            "AnyHealthcare": t[2].toggle("Has insurance", True),
            "NoDocbcCost": t[3].toggle("Skipped doctor (cost)"),
        }
        go_btn = st.form_submit_button("Predict", type="primary")
    if go_btn:
        row = {**{k_: int(v) for k_, v in flags.items()}, "BMI": bmi,
               "Diabetes": list(DIA_LBL.values()).index(dia),
               "GenHlth": list(GEN_LBL.values()).index(gen) + 1,
               "MentHlth": ment, "PhysHlth": phys, "Sex": SEX_ORDER.index(sex),
               "Age": AGE_ORDER.index(age) + 1,
               "Education": list(EDU_LBL.values()).index(edu) + 1,
               "Income": list(INC_LBL.values()).index(inc) + 1}
        pr = float(model.predict_proba(pd.DataFrame([row])[meta["features"]])[:, 1][0])
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=pr * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ef4444" if pr >= THR else "#22c55e"},
                   "threshold": {"line": {"color": "#f59e0b", "width": 4}, "value": THR * 100}}))
        st.plotly_chart(sty(g, 200), width="stretch")
        (st.error if pr >= THR else st.success)(
            "High-risk group (above threshold)" if pr >= THR else "Below the risk threshold")
        st.caption("Screening model, not a medical diagnosis.")

# ---------------------------------------------------------------- charts
st.markdown("### Cardiovascular risk factors")
r1a, r1b = st.columns(2)

with r1a.container(border=True):
    st.markdown("**Risk by Age Group** · click a bar to cross-filter")
    d = by_sex(base)
    rate = d.groupby("Age_lbl", observed=False)[TARGET].mean().reindex(AGE_ORDER) * 100
    colors = ["#ef4444" if (not age_sel or a_ in age_sel) else "#4b5563" for a_ in AGE_ORDER]
    fig = go.Figure(go.Bar(x=AGE_ORDER, y=rate.values, marker_color=colors, name="CVD rate"))
    fig.update_layout(yaxis_title="CVD rate (%)", xaxis_title="Age group", showlegend=False)
    st.plotly_chart(sty(fig), key=AGE_KEY, on_select="rerun", selection_mode="points",
                    width="stretch")

with r1b.container(border=True):
    st.markdown("**Risk by Sex** · share of CVD cases · click a slice")
    d = by_age(base)
    cases = d[d[TARGET] == 1].Sex_lbl.value_counts().reindex(SEX_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=SEX_ORDER, values=cases.values, hole=0.55, sort=False,
                           marker_colors=["#f472b6", "#38bdf8"]))
    fig.update_layout(legend_title_text="Sex")
    st.plotly_chart(sty(fig), key=SEX_KEY, on_select="rerun", selection_mode="points",
                    width="stretch")

r2a, r2b = st.columns(2)
with r2a.container(border=True):
    st.markdown("**High BP vs Cardiovascular Disease**")
    g = (df.dropna(subset=["HighBP"]).assign(BP=lambda x: x.HighBP.map({0: "Normal BP", 1: "High BP"}))
         .groupby(["Age_lbl", "BP"], observed=True)[TARGET].mean().mul(100)
         .reset_index(name="CVD rate (%)"))
    fig = px.line(g, x="Age_lbl", y="CVD rate (%)", color="BP", markers=True,
                  labels={"Age_lbl": "Age group"},
                  color_discrete_map={"High BP": "#ef4444", "Normal BP": "#38bdf8"})
    fig.update_layout(legend_title_text="Blood pressure")
    st.plotly_chart(sty(fig), width="stretch")

with r2b.container(border=True):
    st.markdown("**BMI and Cardiovascular Risk**")
    t1, t2 = st.tabs(["Distribution", "Box plot"])
    dd = df.dropna(subset=["BMI"])
    with t1:
        fig = px.histogram(dd, x="BMI", color="CVD", nbins=40, barmode="overlay",
                           histnorm="percent", opacity=0.65,
                           color_discrete_map={"CVD": "#ef4444", "No CVD": "#38bdf8"})
        fig.update_layout(legend_title_text="Status", yaxis_title="% of group")
        st.plotly_chart(sty(fig, 290), width="stretch")
    with t2:
        fig = px.box(dd, x="CVD", y="BMI", color="CVD",
                     color_discrete_map={"CVD": "#ef4444", "No CVD": "#38bdf8"})
        fig.update_layout(legend_title_text="Status")
        st.plotly_chart(sty(fig, 290), width="stretch")

r3a, r3b = st.columns(2)
with r3a.container(border=True):
    st.markdown("**Health Condition Burden**")
    t1, t2 = st.tabs(["Conditions", "BMI vs risk"])
    with t1:
        conds = {"Stroke": df.Stroke, "High chol.": df.HighChol, "High BP": df.HighBP,
                 "Hard to walk": df.DiffWalk, "Diabetes": df.Diabetes.map({0: 0, 1: 0, 2: 1})}
        st.plotly_chart(yn_bar(df, conds, ""), width="stretch")
    with t2:
        s = (df.dropna(subset=["GenHlth", "BMI"]).assign(GH=lambda x: x.GenHlth.map(GEN_LBL))
             .groupby(["GH", "Age_lbl"], observed=True)
             .agg(BMI=("BMI", "mean"), Rate=(TARGET, "mean"), Records=(TARGET, "size"))
             .reset_index())
        s["Rate"] *= 100
        fig = px.scatter(s, x="BMI", y="Rate", size="Records", color="GH",
                         hover_name="Age_lbl", labels={"Rate": "CVD rate (%)", "GH": "General health"},
                         category_orders={"GH": list(GEN_LBL.values())})
        st.plotly_chart(sty(fig, 290), width="stretch")

with r3b.container(border=True):
    st.markdown("**Lifestyle Risk Factors**")
    life = {"Smoker": df.Smoker, "Active": df.PhysActivity, "Fruits": df.Fruits,
            "Veggies": df.Veggies, "Heavy alcohol": df.HvyAlcoholConsump}
    st.plotly_chart(yn_bar(df, life, ""), width="stretch")

r4a, r4b = st.columns(2)
with r4a.container(border=True):
    st.markdown("**Socioeconomic & Healthcare**")
    t1, t2 = st.tabs(["Treemap (click to drill down)", "Healthcare access"])
    with t1:
        tm = (df.groupby(["Inc_lbl", "Edu_lbl"], observed=True)
              .agg(Records=(TARGET, "size"), Rate=(TARGET, "mean")).reset_index())
        tm["Rate"] *= 100
        fig = px.treemap(tm, path=[px.Constant("All"), "Inc_lbl", "Edu_lbl"], values="Records",
                         color="Rate", color_continuous_scale="RdYlGn_r",
                         labels={"Rate": "CVD rate (%)"})
        st.plotly_chart(sty(fig, 290), width="stretch")
    with t2:
        acc = {"Has insurance": df.AnyHealthcare, "Skipped doctor (cost)": df.NoDocbcCost}
        st.plotly_chart(yn_bar(df, acc, ""), width="stretch")

with r4b.container(border=True):
    st.markdown("**Risk Factor Interactions** · High BP × High cholesterol")
    h = df.dropna(subset=["HighBP", "HighChol"])
    piv = (h.groupby(["HighBP", "HighChol"])[TARGET].mean().mul(100).unstack()
           .rename(index={0: "Normal BP", 1: "High BP"}, columns={0: "Normal chol.", 1: "High chol."}))
    fig = px.imshow(piv, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                    labels={"color": "CVD rate (%)"})
    st.plotly_chart(sty(fig), width="stretch")

with st.container(border=True):
    st.markdown("**Geographic map**")
    st.info("Placeholder: the dataset has no state/location column. "
            "Add geographic data to enable the map.")

# ---------------------------------------------------------------- summary + footer
st.markdown("### Filtered Population Summary")
st.caption("Aggregated statistics only; no individual records are shown.")
s = st.columns(4)
s[0].metric("Records", f"{len(df):,}")
s[1].metric("CVD rate", f"{df[TARGET].mean() * 100:.1f}%")
s[2].metric("Average BMI", f"{df.BMI.mean():.1f}")
s[3].metric("High BP", f"{df.HighBP.mean() * 100:.1f}%")

m = meta["test_metrics"]
st.caption(f"CardioView Analytics · Data is de-identified and intended for analytical use only. "
           f"· Model: {meta['model_name']} · ROC-AUC {m['roc_auc']:.3f} · "
           f"PR-AUC {m['pr_auc']:.3f} · Recall {m['recall']:.2f} · Precision {m['precision']:.2f}")