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

st.set_page_config(page_title="Bảng điều khiển nguy cơ tim mạch", layout="wide",
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
EDU_LBL = {1: "Chưa đi học", 2: "Tiểu học", 3: "THPT dở dang", 4: "Tốt nghiệp THPT",
           5: "Đại học dở dang", 6: "Tốt nghiệp đại học"}
EDU_BINS = {"THPT trở xuống": [1, 2, 3, 4], "Đại học dở dang": [5],
            "Tốt nghiệp đại học": [6]}
DIA_LBL = {0: "Không", 1: "Tiền tiểu đường", 2: "Có"}
GEN_LBL = {1: "Xuất sắc", 2: "Rất tốt", 3: "Tốt", 4: "Trung bình", 5: "Kém"}
SEX_ORDER = ["Nữ", "Nam"]


@st.cache_data
def load():
    d = pd.read_csv(CSV)
    d = d[d.BMI.isna() | d.BMI.between(12, 60)].reset_index(drop=True)
    d["Age_lbl"] = pd.Categorical(d.Age.map(AGE_LBL), AGE_ORDER, ordered=True)
    d["Sex_lbl"] = d.Sex.map({0: "Nữ", 1: "Nam"})
    d["Inc_lbl"] = pd.Categorical(d.Income.map(INC_LBL), list(INC_LBL.values()), ordered=True)
    d["Edu_lbl"] = pd.Categorical(d.Education.map(EDU_LBL), list(EDU_LBL.values()), ordered=True)
    d["Bệnh tim"] = d[TARGET].map({0: "Không bệnh tim", 1: "Bệnh tim"})
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
DEFAULTS = {"f_sex": "Tất cả giới tính", "f_age": "Tất cả nhóm tuổi", "f_bp": "Tất cả",
            "f_chol": "Tất cả", "f_dia": "Tất cả", "f_smk": "Tất cả", "f_act": "Tất cả",
            "f_inc": "Tất cả mức thu nhập", "f_edu": "Tất cả trình độ",
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
    st.caption("Sức khỏe dân số")
    st.divider()
    st.markdown("### ⚙️ Bộ lọc chung")
    st.selectbox("Giới tính", ["Tất cả giới tính", "Nữ", "Nam"], key="f_sex")
    st.selectbox("Nhóm tuổi", ["Tất cả nhóm tuổi"] + list(AGE_BINS), key="f_age")
    st.selectbox("Huyết áp cao", ["Tất cả", "Không", "Có"], key="f_bp")
    st.selectbox("Cholesterol cao", ["Tất cả", "Không", "Có"], key="f_chol")
    st.selectbox("Tiểu đường", ["Tất cả", "Không", "Tiền tiểu đường", "Có"], key="f_dia")
    st.selectbox("Hút thuốc", ["Tất cả", "Không", "Có"], key="f_smk")
    st.selectbox("Vận động thể chất", ["Tất cả", "Không", "Có"], key="f_act")
    st.selectbox("Thu nhập", ["Tất cả mức thu nhập"] + list(INC_BINS), key="f_inc")
    st.selectbox("Trình độ học vấn", ["Tất cả trình độ"] + list(EDU_BINS), key="f_edu")
    st.slider("Khoảng BMI", 12.0, 60.0, step=0.5, key="f_bmi")
    st.divider()
    st.button("↻ Đặt lại tất cả", width="stretch", on_click=reset)
    st.caption("ĐỘ PHỦ DỮ LIỆU")


def yn(d, col, v):
    return d if v == "Tất cả" else d[d[col] == (1 if v == "Có" else 0)]


S = st.session_state
base = data
if S.f_sex != "Tất cả giới tính":
    base = base[base.Sex_lbl == S.f_sex]
if S.f_age != "Tất cả nhóm tuổi":
    base = base[base.Age.isin(AGE_BINS[S.f_age])]
if S.f_inc != "Tất cả mức thu nhập":
    base = base[base.Income.isin(INC_BINS[S.f_inc])]
if S.f_edu != "Tất cả trình độ":
    base = base[base.Education.isin(EDU_BINS[S.f_edu])]
if S.f_dia != "Tất cả":
    base = base[base.Diabetes == {"Không": 0, "Tiền tiểu đường": 1, "Có": 2}[S.f_dia]]
base = yn(yn(yn(yn(base, "HighBP", S.f_bp), "HighChol", S.f_chol), "Smoker", S.f_smk),
          "PhysActivity", S.f_act)
base = base[base.BMI.isna() | base.BMI.between(*S.f_bmi)]
with st.sidebar:
    st.progress(len(base) / len(data))
    st.caption(f"{len(base):,} / {len(data):,} bản ghi phù hợp bộ lọc.")


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
        for v, name in [(1, "Có"), (0, "Không")]:
            m = d[TARGET][s.reindex(d.index) == v]
            rows.append((lbl, name, m.mean() * 100 if len(m) else np.nan))
    return pd.DataFrame(rows, columns=["Yếu tố", "Trả lời", "Tỷ lệ bệnh tim (%)"])


def yn_bar(d, cols, title):
    fig = px.bar(rate_yes_no(d, cols), x="Yếu tố", y="Tỷ lệ bệnh tim (%)", color="Trả lời",
                 barmode="group", title=title,
                 color_discrete_map={"Có": "#ef4444", "Không": "#38bdf8"})
    fig.update_layout(legend_title_text="Có yếu tố này?")
    return sty(fig)


# ---------------------------------------------------------------- header
st.markdown('<div class="eyebrow">TỔNG QUAN SỨC KHỎE DÂN SỐ</div>'
            '<div class="page-title">Bảng điều khiển nguy cơ tim mạch</div>'
            '<div class="sub">Khám phá các yếu tố nhân khẩu học, sức khỏe, lối sống, kinh tế - xã hội '
            'và nguy cơ dự đoán. Bấm vào cột hoặc lát cắt để lọc chéo.</div>',
            unsafe_allow_html=True)

if len(df) == 0:
    st.warning("Không có bản ghi nào phù hợp bộ lọc hiện tại.")
    st.stop()

if age_sel or sex_sel:
    c1, c2 = st.columns([5, 1])
    c1.info("Đang lọc chéo: " + " · ".join(age_sel + sex_sel))
    c2.button("Bỏ chọn", on_click=clear_selection, width="stretch")

k = st.columns(4)
k[0].metric("Tổng số bản ghi", f"{len(df):,}")
k[1].metric("Tỷ lệ bệnh tim mạch", f"{df[TARGET].mean() * 100:.1f}%")
k[2].metric("Nhóm tuổi đông nhất", str(df.Age_lbl.value_counts().idxmax()))
k[3].metric("BMI trung bình", f"{df.BMI.mean():.1f}")

# ---------------------------------------------------------------- risk + prediction
st.markdown("### Tổng quan nguy cơ & dự đoán")
left, right = st.columns(2)

with left.container(border=True):
    st.markdown("**Tổng quan nguy cơ dân số**")
    st.caption(f"Nguy cơ do mô hình dự đoán cho nhóm dân số đã lọc "
               f"(ngưỡng {THR:.0%}).")
    p = proba.loc[df.index]
    fig = px.histogram(p * 100, nbins=50, labels={"value": "Nguy cơ dự đoán (%)"})
    fig.add_vline(x=THR * 100, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Ngưỡng")
    fig.update_layout(showlegend=False, yaxis_title="Số người")
    st.plotly_chart(sty(fig, 260), width="stretch")
    st.metric("Nhóm nguy cơ cao", f"{(p >= THR).mean() * 100:.1f}%")

with right.container(border=True):
    st.markdown("**Dự đoán nguy cơ bằng XGBoost**")
    with st.form("predict"):
        a, b, c = st.columns(3)
        sex = a.selectbox("Giới tính", SEX_ORDER)
        age = b.selectbox("Tuổi", AGE_ORDER, index=6)
        bmi = c.number_input("BMI", 12.0, 60.0, 27.0)
        edu = a.selectbox("Trình độ học vấn", list(EDU_LBL.values()), index=5)
        inc = b.selectbox("Thu nhập", list(INC_LBL.values()), index=6)
        dia = c.selectbox("Tiểu đường", list(DIA_LBL.values()))
        gen = a.select_slider("Sức khỏe tổng quát", list(GEN_LBL.values()), "Tốt")
        ment = b.slider("Số ngày tinh thần không tốt (30 ngày qua)", 0, 30, 0)
        phys = c.slider("Số ngày thể chất không tốt (30 ngày qua)", 0, 30, 0)
        t = st.columns(4)
        flags = {
            "HighBP": t[0].toggle("Huyết áp cao"), "HighChol": t[1].toggle("Cholesterol cao"),
            "CholCheck": t[2].toggle("Đã đo cholesterol", True), "Smoker": t[3].toggle("Hút thuốc"),
            "Stroke": t[0].toggle("Đột quỵ"), "DiffWalk": t[1].toggle("Khó đi lại"),
            "PhysActivity": t[2].toggle("Vận động thể chất", True), "Fruits": t[3].toggle("Ăn trái cây"),
            "Veggies": t[0].toggle("Ăn rau", True), "HvyAlcoholConsump": t[1].toggle("Uống rượu nhiều"),
            "AnyHealthcare": t[2].toggle("Có bảo hiểm y tế", True),
            "NoDocbcCost": t[3].toggle("Bỏ khám vì chi phí"),
        }
        go_btn = st.form_submit_button("Dự đoán", type="primary")
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
            "Nhóm nguy cơ cao (vượt ngưỡng)" if pr >= THR else "Dưới ngưỡng nguy cơ")
        st.caption("Mô hình sàng lọc, không phải chẩn đoán y khoa.")

# ---------------------------------------------------------------- charts
st.markdown("### Các yếu tố nguy cơ tim mạch")
r1a, r1b = st.columns(2)

with r1a.container(border=True):
    st.markdown("**Nguy cơ theo nhóm tuổi** · bấm vào cột để lọc chéo")
    d = by_sex(base)
    rate = d.groupby("Age_lbl", observed=False)[TARGET].mean().reindex(AGE_ORDER) * 100
    colors = ["#ef4444" if (not age_sel or a_ in age_sel) else "#4b5563" for a_ in AGE_ORDER]
    fig = go.Figure(go.Bar(x=AGE_ORDER, y=rate.values, marker_color=colors, name="Tỷ lệ bệnh tim"))
    fig.update_layout(yaxis_title="Tỷ lệ bệnh tim (%)", xaxis_title="Nhóm tuổi", showlegend=False)
    st.plotly_chart(sty(fig), key=AGE_KEY, on_select="rerun", selection_mode="points",
                    width="stretch")

with r1b.container(border=True):
    st.markdown("**Nguy cơ theo giới tính** · tỷ trọng ca bệnh tim · bấm vào lát cắt")
    d = by_age(base)
    cases = d[d[TARGET] == 1].Sex_lbl.value_counts().reindex(SEX_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=SEX_ORDER, values=cases.values, hole=0.55, sort=False,
                           marker_colors=["#f472b6", "#38bdf8"]))
    fig.update_layout(legend_title_text="Giới tính")
    st.plotly_chart(sty(fig), key=SEX_KEY, on_select="rerun", selection_mode="points",
                    width="stretch")

r2a, r2b = st.columns(2)
with r2a.container(border=True):
    st.markdown("**Huyết áp cao và bệnh tim mạch**")
    g = (df.dropna(subset=["HighBP"]).assign(BP=lambda x: x.HighBP.map({0: "Huyết áp bình thường", 1: "Huyết áp cao"}))
         .groupby(["Age_lbl", "BP"], observed=True)[TARGET].mean().mul(100)
         .reset_index(name="Tỷ lệ bệnh tim (%)"))
    fig = px.line(g, x="Age_lbl", y="Tỷ lệ bệnh tim (%)", color="BP", markers=True,
                  labels={"Age_lbl": "Nhóm tuổi"},
                  color_discrete_map={"Huyết áp cao": "#ef4444", "Huyết áp bình thường": "#38bdf8"})
    fig.update_layout(legend_title_text="Huyết áp")
    st.plotly_chart(sty(fig), width="stretch")

with r2b.container(border=True):
    st.markdown("**BMI và nguy cơ tim mạch**")
    t1, t2 = st.tabs(["Phân bố", "Biểu đồ hộp"])
    dd = df.dropna(subset=["BMI"])
    with t1:
        fig = px.histogram(dd, x="BMI", color="Bệnh tim", nbins=40, barmode="overlay",
                           histnorm="percent", opacity=0.65,
                           color_discrete_map={"Bệnh tim": "#ef4444", "Không bệnh tim": "#38bdf8"})
        fig.update_layout(legend_title_text="Nhóm", yaxis_title="% trong nhóm")
        st.plotly_chart(sty(fig, 290), width="stretch")
    with t2:
        fig = px.box(dd, x="Bệnh tim", y="BMI", color="Bệnh tim",
                     color_discrete_map={"Bệnh tim": "#ef4444", "Không bệnh tim": "#38bdf8"})
        fig.update_layout(legend_title_text="Nhóm")
        st.plotly_chart(sty(fig, 290), width="stretch")

r3a, r3b = st.columns(2)
with r3a.container(border=True):
    st.markdown("**Gánh nặng bệnh lý**")
    t1, t2 = st.tabs(["Bệnh lý", "BMI và nguy cơ"])
    with t1:
        conds = {"Đột quỵ": df.Stroke, "Cholesterol cao": df.HighChol, "Huyết áp cao": df.HighBP,
                 "Khó đi lại": df.DiffWalk, "Tiểu đường": df.Diabetes.map({0: 0, 1: 0, 2: 1})}
        st.plotly_chart(yn_bar(df, conds, ""), width="stretch")
    with t2:
        s = (df.dropna(subset=["GenHlth", "BMI"]).assign(GH=lambda x: x.GenHlth.map(GEN_LBL))
             .groupby(["GH", "Age_lbl"], observed=True)
             .agg(BMI=("BMI", "mean"), Rate=(TARGET, "mean"), Records=(TARGET, "size"))
             .reset_index())
        s["Rate"] *= 100
        fig = px.scatter(s, x="BMI", y="Rate", size="Records", color="GH",
                         hover_name="Age_lbl", labels={"Rate": "Tỷ lệ bệnh tim (%)", "GH": "Sức khỏe tổng quát", "Records": "Số bản ghi"},
                         category_orders={"GH": list(GEN_LBL.values())})
        st.plotly_chart(sty(fig, 290), width="stretch")

with r3b.container(border=True):
    st.markdown("**Yếu tố lối sống**")
    life = {"Hút thuốc": df.Smoker, "Vận động": df.PhysActivity, "Trái cây": df.Fruits,
            "Rau": df.Veggies, "Rượu nhiều": df.HvyAlcoholConsump}
    st.plotly_chart(yn_bar(df, life, ""), width="stretch")

r4a, r4b = st.columns(2)
with r4a.container(border=True):
    st.markdown("**Kinh tế - xã hội & chăm sóc y tế**")
    t1, t2 = st.tabs(["Treemap (bấm để xem chi tiết)", "Tiếp cận y tế"])
    with t1:
        tm = (df.groupby(["Inc_lbl", "Edu_lbl"], observed=True)
              .agg(Records=(TARGET, "size"), Rate=(TARGET, "mean")).reset_index())
        tm["Rate"] *= 100
        fig = px.treemap(tm, path=[px.Constant("Tất cả"), "Inc_lbl", "Edu_lbl"], values="Records",
                         color="Rate", color_continuous_scale="RdYlGn_r",
                         labels={"Rate": "Tỷ lệ bệnh tim (%)", "Records": "Số bản ghi", "Inc_lbl": "Thu nhập", "Edu_lbl": "Học vấn"})
        st.plotly_chart(sty(fig, 290), width="stretch")
    with t2:
        acc = {"Có bảo hiểm": df.AnyHealthcare, "Bỏ khám vì chi phí": df.NoDocbcCost}
        st.plotly_chart(yn_bar(df, acc, ""), width="stretch")

with r4b.container(border=True):
    st.markdown("**Tương tác yếu tố nguy cơ** · Huyết áp cao × Cholesterol cao")
    h = df.dropna(subset=["HighBP", "HighChol"])
    piv = (h.groupby(["HighBP", "HighChol"])[TARGET].mean().mul(100).unstack()
           .rename(index={0: "Huyết áp bình thường", 1: "Huyết áp cao"}, columns={0: "Cholesterol bình thường", 1: "Cholesterol cao"}))
    fig = px.imshow(piv, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                    labels={"color": "Tỷ lệ bệnh tim (%)"})
    st.plotly_chart(sty(fig), width="stretch")

with st.container(border=True):
    st.markdown("**Bản đồ địa lý**")
    st.info("Chỗ giữ chỗ: bộ dữ liệu chưa có cột bang/vị trí. "
            "Hãy bổ sung dữ liệu địa lý để bật bản đồ.")

# ---------------------------------------------------------------- summary + footer
st.markdown("### Tóm tắt nhóm dân số đã lọc")
st.caption("Chỉ hiển thị thống kê tổng hợp, không hiển thị bản ghi cá nhân.")
s = st.columns(4)
s[0].metric("Số bản ghi", f"{len(df):,}")
s[1].metric("Tỷ lệ bệnh tim mạch", f"{df[TARGET].mean() * 100:.1f}%")
s[2].metric("BMI trung bình", f"{df.BMI.mean():.1f}")
s[3].metric("Huyết áp cao", f"{df.HighBP.mean() * 100:.1f}%")

m = meta["test_metrics"]
st.caption(f"CardioView Analytics · Dữ liệu đã được ẩn danh và chỉ dùng cho mục đích phân tích. "
           f"· Mô hình: {meta['model_name']} · ROC-AUC {m['roc_auc']:.3f} · "
           f"PR-AUC {m['pr_auc']:.3f} · Recall {m['recall']:.2f} · Precision {m['precision']:.2f}")
