import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# ==============================================================================
# CẤU HÌNH & ĐƯỜNG DẪN
# ==============================================================================
BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "final" / "heart_disease_health_indicators_BRFSS2015_V2.csv"
MODEL_DIR = BASE / "src" / "model" / "Logisic Regression" / "model_detail"
MODEL = MODEL_DIR / "logistic_regression_final.pkl"
META = MODEL_DIR / "model_metadata.json"
TARGET = "HeartDiseaseorAttack"

st.set_page_config(
    page_title="CardioView - Bảng Điều Khiển Nguy Cơ Tim Mạch",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1550px; padding-top: 4rem; padding-bottom: 3rem; }
    .eyebrow { font-size: 1.5rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #ef4444; margin-bottom: 0.2rem; }
    .page-title { font-size: 2.1rem; font-weight: 800; color: #f3f4f6; margin-bottom: 0.2rem; }
    .sub { color: #9ca3af; font-size: 0.92rem; margin-bottom: 1.2rem; line-height: 1.5; }
    [data-testid=stMetric] { background: rgba(31, 41, 55, 0.6); border: 1px solid #374151;
        border-radius: 12px; padding: 0.9rem 1.1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# NHÃN & TỪ ĐIỂN
# ==============================================================================
AGE_ORDER = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
             "55–59", "60–64", "65–69", "70–74", "75–79", "80+"]
AGE_LBL = dict(enumerate(AGE_ORDER, 1))
AGE_BINS = {"18–24": [1], "25–34": [2, 3], "35–44": [4, 5], "45–54": [6, 7],
            "55–64": [8, 9], "65–74": [10, 11], "75+": [12, 13]}
INC_LBL = {1: "<$10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
           5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: "$75k+"}
INC_BINS = {"< $25k": [1, 2, 3, 4], "$25–50k": [5, 6], "$50–75k": [7], "$75k+": [8]}
EDU_LBL = {1: "Chưa đi học", 2: "Tiểu học", 3: "THPT dở dang",
           4: "Tốt nghiệp THPT", 5: "Đại học dở dang", 6: "Tốt nghiệp đại học"}
EDU_BINS = {"THPT trở xuống": [1, 2, 3, 4], "Đại học dở dang": [5], "Tốt nghiệp đại học": [6]}
DIA_LBL = {0: "Không", 1: "Tiền tiểu đường", 2: "Có"}
GEN_LBL = {1: "Xuất sắc", 2: "Rất tốt", 3: "Tốt", 4: "Trung bình", 5: "Kém"}
SEX_ORDER = ["Nữ", "Nam"]

LBL_POS = "Có tiền sử bệnh tim/đau tim"
LBL_NEG = "Không có tiền sử"
RATE = "Tỷ lệ có tiền sử bệnh tim/đau tim (%)"
ALL = "Tất cả"
BP_N, BP_Y = "Huyết áp bình thường", "Huyết áp cao"

STATE_CODE_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI',
    'Wyoming': 'WY', 'Puerto Rico': 'PR', 'Guam': 'GU'
}
CODE_TO_STATE = {v: k for k, v in STATE_CODE_MAP.items()}
ALL_STATES = sorted(STATE_CODE_MAP)


# ==============================================================================
# NẠP DỮ LIỆU & MÔ HÌNH
# ==============================================================================
@st.cache_resource
def get_model():
    md = joblib.load(MODEL)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    return md["model"], md["scaler"], md["features"], md["threshold"], meta


@st.cache_resource
def load():
    _, _, features, _, _ = get_model()
    d = pd.read_csv(CSV)
    d[TARGET] = d[TARGET].fillna(0).astype(int)
    # Mã hóa giới tính: file gốc 1=Nam, 2=Nữ; mô hình được huấn luyện với 1=Nam, 0=Nữ
    if d.Sex.max() == 2:
        d["Sex"] = (d.Sex == 1).astype(int)
    # Mô hình chỉ huấn luyện/đánh giá trên các hồ sơ ĐẦY ĐỦ dữ liệu. Tái tạo đúng tập Test
    # (80/20, random_state=42, stratify) trên các dòng đó, trước mọi bước lọc khác.
    d["is_complete"] = d[features + [TARGET]].notna().all(axis=1)
    idx = np.flatnonzero(d["is_complete"].values)
    _, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=d[TARGET].values[idx])
    d["is_test"] = np.isin(np.arange(len(d)), te)
    d["Age_lbl"] = pd.Categorical(d.Age.map(AGE_LBL), AGE_ORDER, ordered=True)
    d["Sex_lbl"] = d.Sex.map({0: "Nữ", 1: "Nam"})
    d["Inc_lbl"] = pd.Categorical(d.Income.map(INC_LBL), list(INC_LBL.values()), ordered=True)
    d["Edu_lbl"] = pd.Categorical(d.Education.map(EDU_LBL), list(EDU_LBL.values()), ordered=True)
    d["Bệnh tim"] = d[TARGET].map({0: LBL_NEG, 1: LBL_POS})
    return d


@st.cache_resource
def predict_all():
    # Chỉ chấm điểm các hồ sơ đầy đủ (Logistic Regression không nhận giá trị thiếu)
    model, scaler, features, _, _ = get_model()
    d = load()
    out = pd.Series(np.nan, index=d.index)
    c = d[d.is_complete]
    out.loc[c.index] = model.predict_proba(scaler.transform(c[features]))[:, 1]
    return out


data = load()
model, scaler, features, THR, meta = get_model()
proba = predict_all()
NATIONAL_RATE = data[TARGET].mean() * 100
N_TEST = int(data.is_test.sum())
N_COMPLETE = int(data.is_complete.sum())
EVAL_ALL = "Toàn bộ dữ liệu dân số"
EVAL_TEST = f"Chỉ tập Test độc lập ({N_TEST:,} mẫu)"

# ==============================================================================
# THANH BÊN: BỘ LỌC ĐA CẤP
# ==============================================================================
DEFAULTS = {
    "f_state": "Tất cả các bang", "f_sex": "Tất cả giới tính", "f_age": "Tất cả nhóm tuổi",
    "f_bp": ALL, "f_chol": ALL, "f_dia": ALL, "f_smk": ALL, "f_act": ALL,
    "f_inc": "Tất cả mức thu nhập", "f_edu": "Tất cả trình độ",
    "f_bmi": (12.0, 60.0), "f_eval_set": EVAL_ALL, "ver": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset():
    ver = st.session_state.ver + 1
    st.session_state.update(DEFAULTS)
    st.session_state.ver = ver


def clear_cross_selection():
    st.session_state.ver += 1


def reset_state_filter():
    st.session_state.f_state = "Tất cả các bang"
    st.session_state.ver += 1


def on_map_select():
    # Chạy TRƯỚC khi script chạy lại nên được phép đổi giá trị ô chọn bang ở thanh bên
    try:
        pts = st.session_state[f"map_{st.session_state.ver}"]["selection"]["points"]
    except (KeyError, TypeError):
        return
    if not pts:
        return
    p = pts[0]
    name = CODE_TO_STATE.get(p.get("location"))
    pn = p.get("point_number")
    if name is None and pn is not None and pn < len(ALL_STATES):
        name = ALL_STATES[pn]
    if name:
        st.session_state.f_state = name


with st.sidebar:
    st.markdown("## ❤️ CardioView")
    st.caption("Hệ thống Giám sát & Dự đoán Sức khỏe Dân số")
    st.divider()

    st.markdown("### 🗺️ Cấp 1: Phân bố Địa lý")
    st.selectbox("Bang / Vùng lãnh thổ", ["Tất cả các bang"] + ALL_STATES, key="f_state")

    st.markdown("### 👤 Cấp 2: Nhân khẩu học")
    st.selectbox("Giới tính", ["Tất cả giới tính", "Nữ", "Nam"], key="f_sex")
    st.selectbox("Nhóm tuổi", ["Tất cả nhóm tuổi"] + list(AGE_BINS), key="f_age")
    st.selectbox("Thu nhập bình quân", ["Tất cả mức thu nhập"] + list(INC_BINS), key="f_inc")
    st.selectbox("Trình độ học vấn", ["Tất cả trình độ"] + list(EDU_BINS), key="f_edu")

    st.markdown("### 🩺 Cấp 3: Tiền sử Lâm sàng")
    st.selectbox("Huyết áp cao", [ALL, "Không", "Có"], key="f_bp")
    st.selectbox("Cholesterol cao", [ALL, "Không", "Có"], key="f_chol")
    st.selectbox("Tiểu đường", [ALL, "Không", "Tiền tiểu đường", "Có"], key="f_dia")

    st.markdown("### 🏃 Cấp 4: Lối sống & Thể trạng")
    st.selectbox("Hút thuốc", [ALL, "Không", "Có"], key="f_smk")
    st.selectbox("Vận động thể chất", [ALL, "Không", "Có"], key="f_act")
    st.slider("Chỉ số khối cơ thể (BMI)", 12.0, 60.0, step=0.5, key="f_bmi",
              help="Chỉ hiển thị BMI trong khoảng sinh học hợp lý 12 - 60 kg/m²")

    st.markdown("### 🧪 Phân chia Dữ liệu Đánh giá")
    st.radio("Tập dữ liệu hiển thị:", [EVAL_ALL, EVAL_TEST], key="f_eval_set",
             help="Tập Test được tái tạo đúng như lúc huấn luyện mô hình (80/20, random_state=42).")

    st.divider()
    st.button("↻ Đặt lại tất cả bộ lọc", width="stretch", on_click=reset)
    st.caption("TIẾN TRÌNH ĐỘ PHỦ DỮ LIỆU")


def yn(d, col, v):
    return d if v == ALL else d[d[col] == (1 if v == "Có" else 0)]


S = st.session_state
base_data = data[data.is_test] if S.f_eval_set == EVAL_TEST else data

# base_geo: áp dụng mọi bộ lọc TRỪ bang, để bản đồ và Top 5 không mất ngữ cảnh toàn quốc
base_geo = base_data
if S.f_sex != "Tất cả giới tính":
    base_geo = base_geo[base_geo.Sex_lbl == S.f_sex]
if S.f_age != "Tất cả nhóm tuổi":
    base_geo = base_geo[base_geo.Age.isin(AGE_BINS[S.f_age])]
if S.f_inc != "Tất cả mức thu nhập":
    base_geo = base_geo[base_geo.Income.isin(INC_BINS[S.f_inc])]
if S.f_edu != "Tất cả trình độ":
    base_geo = base_geo[base_geo.Education.isin(EDU_BINS[S.f_edu])]
if S.f_dia != ALL:
    base_geo = base_geo[base_geo.Diabetes == {"Không": 0, "Tiền tiểu đường": 1, "Có": 2}[S.f_dia]]
base_geo = yn(yn(yn(yn(base_geo, "HighBP", S.f_bp), "HighChol", S.f_chol), "Smoker", S.f_smk),
              "PhysActivity", S.f_act)
base_geo = base_geo[base_geo.BMI.isna() | base_geo.BMI.between(*S.f_bmi)]

base = base_geo
if S.f_state != "Tất cả các bang":
    base = base[base.StateName == S.f_state]

with st.sidebar:
    cov_ratio = len(base) / max(len(base_data), 1)
    st.progress(cov_ratio)
    st.caption(f"**{len(base):,} / {len(base_data):,}** bản ghi phù hợp ({cov_ratio * 100:.1f}% tập dữ liệu).")


# ==============================================================================
# LỌC CHÉO GIỮA CÁC BIỂU ĐỒ
# ==============================================================================
def picked(key, order):
    ev = st.session_state.get(key)
    if not ev or "selection" not in ev or not ev["selection"].get("points"):
        return []
    return [order[p["point_number"]] for p in ev["selection"]["points"]
            if p.get("point_number") is not None and p["point_number"] < len(order)]


AGE_KEY, SEX_KEY, MAP_KEY = f"age_{S.ver}", f"sex_{S.ver}", f"map_{S.ver}"
age_sel = picked(AGE_KEY, AGE_ORDER)
sex_sel = picked(SEX_KEY, SEX_ORDER)


def by_age(d):
    return d[d.Age_lbl.isin(age_sel)] if age_sel else d


def by_sex(d):
    return d[d.Sex_lbl.isin(sex_sel)] if sex_sel else d


df = by_sex(by_age(base))


def sty(fig, h=350):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=15, r=15, t=35, b=15), font=dict(family="sans-serif", size=12),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    return fig


def rate_yes_no(d, cols):
    rows = []
    for lbl, s in cols.items():
        for v, name in [(1, "Có"), (0, "Không")]:
            m = d[TARGET][s == v]
            rows.append((lbl, name, m.mean() * 100 if len(m) else np.nan))
    return pd.DataFrame(rows, columns=["Yếu tố", "Tình trạng", RATE])


def yn_bar(d, cols, title):
    fig = px.bar(rate_yes_no(d, cols), x="Yếu tố", y=RATE, color="Tình trạng", barmode="group",
                 title=title, color_discrete_map={"Có": "#ef4444", "Không": "#38bdf8"}, text_auto=".1f")
    fig.update_layout(legend_title_text="Tình trạng")
    return sty(fig, 320)


# ==============================================================================
# TIÊU ĐỀ & KPI
# ==============================================================================
st.markdown('<div class="eyebrow">HỆ THỐNG TRỰC QUAN HÓA THÔNG TIN Y TẾ & SỨC KHỎE DÂN SỐ</div>',
            unsafe_allow_html=True)
st.markdown('<div class="page-title">Bảng Điều Khiển Phân Tích & Dự Đoán Nguy Cơ Tim Mạch</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Khám phá phân bố địa lý, đặc điểm nhân khẩu học, tiền sử sức khỏe và lối sống; '
    'đồng thời đánh giá mô hình Logistic Regression và ước tính nguy cơ cá nhân.</div>',
    unsafe_allow_html=True,
)

active_cross = []
if age_sel:
    active_cross.append(f"Nhóm tuổi: {', '.join(age_sel)}")
if sex_sel:
    active_cross.append(f"Giới tính: {', '.join(sex_sel)}")
if active_cross or S.f_state != "Tất cả các bang":
    c1, c2, c3 = st.columns([4, 1, 1])
    status = []
    if S.f_state != "Tất cả các bang":
        status.append(f"📍 Bang: **{S.f_state}**")
    if active_cross:
        status.append(f"🔍 Lọc chéo: **{' | '.join(active_cross)}**")
    c1.info(" · ".join(status))
    if active_cross:
        c2.button("✕ Bỏ lọc chéo", on_click=clear_cross_selection, width="stretch")
    if S.f_state != "Tất cả các bang":
        c3.button("🌐 Xem toàn quốc", on_click=reset_state_filter, width="stretch")

if len(df) == 0:
    st.warning("⚠️ Không có bản ghi nào phù hợp với điều kiện bộ lọc hiện tại. Vui lòng nới lỏng tiêu chí lọc.")
    st.stop()

k = st.columns(5)
curr_rate = df[TARGET].mean() * 100
with k[0].container(height=165):
    st.metric("Tổng số bản ghi", f"{len(df):,}", help="Số người tham gia khảo sát phù hợp bộ lọc")
with k[1].container(height=165):
    st.metric("Tỷ lệ có tiền sử bệnh tim/đau tim", f"{curr_rate:.1f}%",
              delta=f"{curr_rate - NATIONAL_RATE:+.1f}% so với toàn quốc", delta_color="inverse",
              help="Tỷ lệ người tham gia khảo sát có tiền sử bệnh tim hoặc từng đau tim trong mẫu")
with k[2].container(height=165):
    st.metric("Nhóm tuổi chiếm ưu thế", str(df.Age_lbl.value_counts().idxmax()),
              help="Nhóm độ tuổi có số lượng người đông nhất")
with k[3].container(height=165):
    st.metric("Chỉ số BMI trung bình", f"{df.BMI.mean():.1f}", help="Chỉ số khối cơ thể trung bình của nhóm")
with k[4].container(height=165):
    st.metric("Tỷ lệ tăng huyết áp", f"{df.HighBP.mean() * 100:.1f}%", help="Tỷ lệ người có tiền sử huyết áp cao")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 1: BẢN ĐỒ ĐỊA LÝ
# ==============================================================================
with st.container(border=True):
    st.markdown("### 1. 🗺️ Bản đồ Địa lý: Phân bố Không gian Nguy cơ Tim mạch `[Click để chọn bang]`")
    st.caption("Bản đồ Choropleth thể hiện tỷ lệ người tham gia có tiền sử bệnh tim/đau tim trên 53 bang/vùng lãnh "
               "thổ của Hoa Kỳ. Bấm vào một bang để lọc dữ liệu. Guam và Puerto Rico có trong dữ liệu nhưng "
               "không hiển thị trên bản đồ.")

    state_agg = (base_geo.groupby("StateName", observed=True)[TARGET]
                 .agg(TongSo="count", SoCa="sum", TyLe="mean")
                 .reindex(ALL_STATES).rename_axis("StateName").reset_index())
    state_agg[["TongSo", "SoCa"]] = state_agg[["TongSo", "SoCa"]].fillna(0).astype(int)
    state_agg["TyLe_pct"] = state_agg["TyLe"] * 100
    state_agg["Code"] = state_agg["StateName"].map(STATE_CODE_MAP)

    map_col, drill_col = st.columns([3, 1])
    with map_col:
        fig_map = px.choropleth(
            state_agg, locations="Code", locationmode="USA-states", color="TyLe_pct", scope="usa",
            hover_name="StateName",
            hover_data={"Code": False, "TyLe_pct": ":.2f", "TongSo": ":,", "SoCa": ":,"},
            color_continuous_scale="Reds",
            labels={"TyLe_pct": RATE, "TongSo": "Số mẫu khảo sát", "SoCa": "Số trường hợp có tiền sử"},
        )
        fig_map.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(255,255,255,0.05)"),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0), height=430,
            coloraxis_colorbar=dict(title="Tỷ lệ (%)", thickness=14, len=0.7),
        )
        st.plotly_chart(fig_map, key=MAP_KEY, on_select=on_map_select,
                        selection_mode="points", width="stretch")

    with drill_col:
        st.markdown("##### 📍 Xếp hạng & So sánh Bang")
        ranked = state_agg.dropna(subset=["TyLe_pct"])
        tab_hi, tab_lo = st.tabs(["🔴 Top 5 Cao nhất", "🟢 Top 5 Thấp nhất"])
        with tab_hi:
            for _, r in ranked.sort_values("TyLe_pct", ascending=False).head(5).iterrows():
                st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({int(r['SoCa']):,} ca)")
        with tab_lo:
            for _, r in ranked.sort_values("TyLe_pct").head(5).iterrows():
                st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({int(r['SoCa']):,} ca)")
        if S.f_state != "Tất cả các bang":
            st.divider()
            sel = state_agg[state_agg.StateName == S.f_state]
            if len(sel) and pd.notna(sel.iloc[0]["TyLe_pct"]):
                r = sel.iloc[0]
                st.markdown(f"**Bang đang chọn: {S.f_state}**")
                st.metric("Tỷ lệ tại bang", f"{r['TyLe_pct']:.1f}%",
                          delta=f"{r['TyLe_pct'] - NATIONAL_RATE:+.1f}% vs Toàn quốc", delta_color="inverse")
                st.caption(f"Tổng: {int(r['SoCa']):,} ca / {int(r['TongSo']):,} bản ghi")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 2: NHÂN KHẨU HỌC & LỌC CHÉO
# ==============================================================================
st.markdown("### 2. 👥 Nhân khẩu học & Liên kết Lọc chéo (Cross-filtering)")
st.caption("Nhấp chuột vào một cột Nhóm tuổi hoặc lát cắt Giới tính để kích hoạt liên kết lọc chéo giữa các biểu đồ.")
r1a, r1b = st.columns(2)

with r1a.container(border=True):
    st.markdown("**Biểu đồ Cột: Tỷ lệ có tiền sử bệnh tim/đau tim theo Nhóm tuổi** `[Cross-filter]`")
    st.caption("Bấm chọn vào cột để lọc chéo theo nhóm tuổi")
    rate_age = by_sex(base).groupby("Age_lbl", observed=False)[TARGET].mean().reindex(AGE_ORDER) * 100
    colors_age = ["#ef4444" if (not age_sel or a_ in age_sel) else "#4b5563" for a_ in AGE_ORDER]
    fig_age = go.Figure(go.Bar(
        x=AGE_ORDER, y=rate_age.values, marker_color=colors_age, name=RATE,
        text=[f"{v:.1f}%" for v in rate_age.values], textposition="outside",
        hovertemplate="<b>Nhóm tuổi %{x}</b><br>Tỷ lệ có tiền sử: %{y:.2f}%<extra></extra>",
    ))
    fig_age.update_layout(yaxis_title=RATE, xaxis_title="Nhóm tuổi", showlegend=False)
    st.plotly_chart(sty(fig_age, 340), key=AGE_KEY, on_select="rerun",
                    selection_mode="points", width="stretch")

with r1b.container(border=True):
    st.markdown("**Biểu đồ Tròn: Cơ cấu ca bệnh theo Giới tính** `[Cross-filter]`")
    d_sex = by_age(base)
    cases_sex = d_sex[d_sex[TARGET] == 1].Sex_lbl.value_counts().reindex(SEX_ORDER).fillna(0).astype(int)
    fem, male = d_sex[d_sex.Sex_lbl == "Nữ"], d_sex[d_sex.Sex_lbl == "Nam"]
    fem_prev = fem[TARGET].mean() * 100 if len(fem) else 0.0
    male_prev = male[TARGET].mean() * 100 if len(male) else 0.0
    st.caption(f"Tỷ lệ mắc trong từng giới: Nam **{male_prev:.1f}%** | Nữ **{fem_prev:.1f}%** "
               f"(Bấm lát cắt để lọc chéo)")
    fig_sex = go.Figure(go.Pie(
        labels=SEX_ORDER, values=cases_sex.values, hole=0.58, sort=False,
        marker=dict(colors=["#f472b6", "#38bdf8"]), textinfo="label+percent",
        hovertemplate="<b>Giới tính: %{label}</b><br>Số ca có tiền sử: %{value:,}"
                      "<br>Tỷ trọng ca bệnh: %{percent}<extra></extra>",
    ))
    fig_sex.update_layout(legend_title_text="Giới tính")
    st.plotly_chart(sty(fig_sex, 340), key=SEX_KEY, on_select="rerun",
                    selection_mode="points", width="stretch")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 3: LÂM SÀNG & BMI
# ==============================================================================
st.markdown("### 3. 🩺 Yếu tố Lâm sàng & Phân tích Sinh trắc học")
r2a, r2b = st.columns(2)

with r2a.container(border=True):
    st.markdown("**Biểu đồ Đường: Xu hướng nguy cơ theo Độ tuổi × Tiền sử Cao huyết áp**")
    st.caption("So sánh tỷ lệ có tiền sử bệnh tim/đau tim theo tuổi giữa người huyết áp cao và bình thường.")
    g_line = (df.dropna(subset=["HighBP"]).assign(BP=lambda x: x.HighBP.map({0: BP_N, 1: BP_Y}))
              .groupby(["Age_lbl", "BP"], observed=True)[TARGET].mean().mul(100).reset_index(name=RATE))
    fig_line = px.line(g_line, x="Age_lbl", y=RATE, color="BP", markers=True,
                       labels={"Age_lbl": "Nhóm tuổi"},
                       color_discrete_map={BP_Y: "#ef4444", BP_N: "#38bdf8"})
    fig_line.update_traces(line=dict(width=3), marker=dict(size=8),
                           hovertemplate="<b>%{x}</b> (%{data.name})<br>Tỷ lệ: %{y:.2f}%<extra></extra>")
    fig_line.update_layout(legend_title_text="Tình trạng Huyết áp",
                           legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                           margin=dict(b=80))
    st.plotly_chart(sty(fig_line, 330), width="stretch")

with r2b.container(border=True):
    st.markdown("**Phân tích Thể trạng: Chỉ số BMI theo Tiền sử bệnh tim/đau tim**")
    tab_dist, tab_box = st.tabs(["📊 Phân bố tần suất (Histogram)", "📦 Biểu đồ Hộp & Điểm phân vị (Box Plot)"])
    dd_bmi = df.dropna(subset=["BMI"])
    cmap = {LBL_POS: "#ef4444", LBL_NEG: "#38bdf8"}
    with tab_dist:
        fig_hist = px.histogram(dd_bmi, x="BMI", color="Bệnh tim", nbins=45, barmode="overlay",
                                histnorm="percent", opacity=0.68, color_discrete_map=cmap)
        fig_hist.update_layout(legend_title_text="Nhóm", yaxis_title="% Tỷ trọng trong nhóm",
                               xaxis_title="Chỉ số khối cơ thể BMI (kg/m²)")
        st.plotly_chart(sty(fig_hist, 290), width="stretch")
    with tab_box:
        fig_box = px.box(dd_bmi, x="Bệnh tim", y="BMI", color="Bệnh tim", points="outliers",
                         color_discrete_map=cmap)
        fig_box.update_layout(legend_title_text="Nhóm", xaxis_title="Tình trạng tiền sử bệnh tim/đau tim",
                              yaxis_title="BMI (kg/m²) [Giới hạn hiển thị 12–60]")
        st.plotly_chart(sty(fig_box, 290), width="stretch")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 4: BỆNH LÝ NỀN & LỐI SỐNG
# ==============================================================================
st.markdown("### 4. 🧬 Gánh nặng Bệnh lý nền & Tương quan Thể trạng")
r3a, r3b = st.columns(2)

with r3a.container(border=True):
    st.markdown("**Gánh nặng Đa bệnh lý & Tương quan Thể trạng**")
    tab_dis, tab_scat = st.tabs(["📊 Các bệnh lý mạn tính liên quan", "🫧 Tương quan Bong bóng (Scatter/Bubble)"])
    with tab_dis:
        conds = {"Đột quỵ": df.Stroke, "Cholesterol cao": df.HighChol, "Huyết áp cao": df.HighBP,
                 "Khó đi lại": df.DiffWalk, "Tiểu đường": df.Diabetes.map({0: 0, 1: 0, 2: 1})}
        st.caption("Tiền tiểu đường được tính là 'Không' trong biểu đồ này.")
        st.plotly_chart(yn_bar(df, conds, "Tỷ lệ có tiền sử bệnh tim/đau tim theo từng yếu tố sức khỏe"),
                        width="stretch")
    with tab_scat:
        s_bubble = (df.dropna(subset=["GenHlth", "BMI"]).assign(GH=lambda x: x.GenHlth.map(GEN_LBL))
                    .groupby(["GH", "Age_lbl"], observed=True)
                    .agg(BMI=("BMI", "mean"), Rate=(TARGET, "mean"), Records=(TARGET, "size")).reset_index())
        s_bubble["Rate"] *= 100
        fig_scat = px.scatter(
            s_bubble, x="BMI", y="Rate", size="Records", color="GH", hover_name="Age_lbl",
            labels={"Rate": RATE, "GH": "Sức khỏe tự đánh giá", "Records": "Số bản ghi", "BMI": "BMI trung bình"},
            category_orders={"GH": list(GEN_LBL.values())},
            title="Tương quan BMI - Tỷ lệ có tiền sử bệnh tim/đau tim theo Nhóm Sức khỏe")
        st.plotly_chart(sty(fig_scat, 320), width="stretch")

with r3b.container(border=True):
    st.markdown("**Yếu tố Thói quen Lối sống**")
    life = {"Hút thuốc": df.Smoker, "Vận động thể chất": df.PhysActivity, "Ăn trái cây": df.Fruits,
            "Ăn rau củ": df.Veggies, "Uống rượu nhiều": df.HvyAlcoholConsump}
    st.plotly_chart(yn_bar(df, life, "Tỷ lệ có tiền sử bệnh tim/đau tim giữa nhóm Có và Không theo lối sống"),
                    width="stretch")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 5: KINH TẾ - XÃ HỘI & TƯƠNG TÁC
# ==============================================================================
st.markdown("### 5. 💰 Kinh tế - Xã hội & Ma trận Tương tác Nguy cơ")
r4a, r4b = st.columns(2)

with r4a.container(border=True):
    st.markdown("**Biểu đồ Cây Phân cấp: Thu nhập × Học vấn (Treemap)** `[Drill-down Phân cấp]`")
    st.caption("Nhấp vào từng khối Thu nhập để đào sâu vào các bậc Học vấn bên trong. "
               "Kích thước = Số bản ghi, Màu = Tỷ lệ có tiền sử bệnh tim/đau tim.")
    tm_df = (df.groupby(["Inc_lbl", "Edu_lbl"], observed=True)
             .agg(Records=(TARGET, "size"), Rate=(TARGET, "mean")).reset_index())
    tm_df["Rate"] *= 100
    fig_tree = px.treemap(
        tm_df, path=[px.Constant("Tất cả mức thu nhập"), "Inc_lbl", "Edu_lbl"], values="Records",
        color="Rate", color_continuous_scale="RdYlGn_r",
        labels={"Rate": RATE, "Records": "Số bản ghi", "Inc_lbl": "Mức thu nhập", "Edu_lbl": "Trình độ học vấn"})
    fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>Số bản ghi: %{value:,}"
                                         "<br>Tỷ lệ: %{color:.2f}%<extra></extra>")
    st.plotly_chart(sty(fig_tree, 340), width="stretch")

with r4b.container(border=True):
    st.markdown("**Bản đồ Nhiệt Ma trận: Tương tác Huyết áp cao × Cholesterol cao**")
    st.caption("So sánh tỷ lệ có tiền sử bệnh tim/đau tim khi đồng thời có huyết áp cao và cholesterol cao.")
    piv = (df.dropna(subset=["HighBP", "HighChol"]).groupby(["HighBP", "HighChol"])[TARGET]
           .mean().mul(100).unstack()
           .rename(index={0: BP_N, 1: BP_Y}, columns={0: "Cholesterol bình thường", 1: "Cholesterol cao"}))
    fig_heat = px.imshow(piv, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                         labels={"color": RATE})
    fig_heat.update_traces(hovertemplate="Huyết áp: %{y}<br>Cholesterol: %{x}<br>Tỷ lệ: %{z:.2f}%<extra></extra>")
    st.plotly_chart(sty(fig_heat, 340), width="stretch")
st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MỤC 6: ĐÁNH GIÁ MÔ HÌNH (TẬP TEST ĐỘC LẬP)
# ==============================================================================
st.markdown("### 6. 🧠 Đánh giá Mô hình Logistic Regression")
st.caption(
    f"Kết quả tính trực tiếp trên tập Test độc lập ({N_TEST:,} mẫu) tại ngưỡng {THR:.0%}. Mô hình chỉ được huấn "
    f"luyện và đánh giá trên {N_COMPLETE:,} hồ sơ đầy đủ dữ liệu; {len(data) - N_COMPLETE:,} hồ sơ thiếu "
    f"cholesterol/BMI không được chấm điểm.")

test_df = data[data.is_test]
y_test = test_df[TARGET]
p_test = proba.loc[test_df.index]
y_pred_test = (p_test >= THR).astype(int)
M = {
    "Accuracy": accuracy_score(y_test, y_pred_test),
    "Precision": precision_score(y_test, y_pred_test),
    "Recall": recall_score(y_test, y_pred_test),
    "F1-score": f1_score(y_test, y_pred_test),
    "ROC-AUC": roc_auc_score(y_test, p_test),
    "PR-AUC": average_precision_score(y_test, p_test),
}
helps = {
    "Accuracy": "Tỷ lệ dự đoán đúng trên tập Test.",
    "Precision": "Trong các trường hợp được dự đoán có bệnh tim, tỷ lệ dự đoán đúng.",
    "Recall": "Trong các trường hợp thực sự có bệnh tim, tỷ lệ được mô hình phát hiện.",
    "F1-score": "Chỉ số cân bằng giữa Precision và Recall.",
    "ROC-AUC": "Khả năng phân biệt hai nhóm ở mọi ngưỡng (1,0 là hoàn hảo, 0,5 là ngẫu nhiên).",
    "PR-AUC": "Diện tích dưới đường Precision-Recall, phù hợp với dữ liệu mất cân bằng.",
}
for col, (name, val) in zip(st.columns(6), M.items()):
    col.metric(name, f"{val:.1%}" if name not in ("ROC-AUC", "PR-AUC") else f"{val:.3f}", help=helps[name])

lbl_actual = y_test.map({0: LBL_NEG, 1: LBL_POS}).values
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**Histogram: Phân phối xác suất dự đoán**")
    hist_df = pd.DataFrame({"Xác suất nguy cơ dự đoán (%)": p_test.values * 100, "Thực tế": lbl_actual})
    fig_hm = px.histogram(hist_df, x="Xác suất nguy cơ dự đoán (%)", color="Thực tế", nbins=40,
                          barmode="overlay", opacity=0.65, histnorm="percent",
                          color_discrete_map={LBL_NEG: "#38bdf8", LBL_POS: "#ef4444"})
    fig_hm.add_vline(x=THR * 100, line_dash="dash", line_color="#f59e0b", line_width=2.5,
                     annotation_text=f"Ngưỡng {THR:.0%}", annotation_position="top right")
    fig_hm.update_layout(yaxis_title="Tỷ trọng (%)", legend_title="Tình trạng thực tế")
    st.plotly_chart(sty(fig_hm, 330), width="stretch")

with c2.container(border=True):
    st.markdown("**Violin Plot: Phân bố điểm nguy cơ theo thực tế**")
    vdf = pd.DataFrame({"Xác suất nguy cơ dự đoán (%)": p_test.values * 100, "Thực tế": lbl_actual})
    fig_vi = px.violin(vdf, x="Thực tế", y="Xác suất nguy cơ dự đoán (%)", color="Thực tế", box=True,
                       points=False, color_discrete_map={LBL_NEG: "#38bdf8", LBL_POS: "#ef4444"})
    fig_vi.add_hline(y=THR * 100, line_dash="dash", line_color="#f59e0b", line_width=2,
                     annotation_text=f"Ngưỡng {THR:.0%}", annotation_position="top right")
    fig_vi.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(sty(fig_vi, 330), width="stretch")

c3, c4 = st.columns(2)
with c3.container(border=True):
    st.markdown(f"**Confusion Matrix tại ngưỡng {THR:.0%}**")
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred_test, labels=[0, 1]),
                         index=["Thực tế: Không bệnh", "Thực tế: Bệnh"],
                         columns=["Dự đoán: Không bệnh", "Dự đoán: Bệnh"])
    fig_cm = px.imshow(cm_df, text_auto=",", color_continuous_scale="Reds", aspect="auto",
                       labels={"color": "Số lượng"})
    fig_cm.update_layout(xaxis_title="Dự đoán", yaxis_title="Thực tế")
    st.plotly_chart(sty(fig_cm, 330), width="stretch")

with c4.container(border=True):
    st.markdown("**Hệ số Logistic Regression: 10 yếu tố tác động mạnh nhất**")
    imp = pd.DataFrame({"Yếu tố": features, "Hệ số": model.coef_[0]})
    imp = imp.reindex(imp["Hệ số"].abs().sort_values().tail(10).index).sort_values("Hệ số")
    fig_imp = px.bar(imp, x="Hệ số", y="Yếu tố", orientation="h", text_auto=".2f",
                     color=imp["Hệ số"] > 0, color_discrete_map={True: "#ef4444", False: "#38bdf8"})
    fig_imp.update_layout(yaxis_title="", xaxis_title="Hệ số (dương: tăng nguy cơ, âm: giảm nguy cơ)",
                          showlegend=False)
    st.plotly_chart(sty(fig_imp, 330), width="stretch")

# ==============================================================================
# MỤC 7: DỰ ĐOÁN NGUY CƠ CÁ NHÂN
# ==============================================================================
st.markdown("### 7. 🩺 Dự đoán nguy cơ cá nhân")
st.caption("Nhập thông tin sức khỏe và lối sống để mô hình Logistic Regression "
           "ước tính xác suất thuộc nhóm có tiền sử bệnh tim/đau tim.")

with st.container(border=True):
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Tình trạng sức khỏe**")
        high_bp = int(st.toggle("Cao huyết áp"))
        high_chol = int(st.toggle("Cholesterol cao"))
        bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=25.0, step=0.1)
        diabetes = st.selectbox("Tiểu đường", [0, 1, 2], format_func=lambda x: DIA_LBL[x])
        gen_hlth = st.selectbox("Tình trạng sức khỏe tổng quát", [1, 2, 3, 4, 5], index=2,
                                format_func=lambda x: GEN_LBL[x])
        age = st.selectbox("Nhóm tuổi", list(AGE_LBL), index=6, format_func=lambda x: AGE_LBL[x])
        ment_hlth = st.slider("Số ngày sức khỏe tinh thần không tốt (30 ngày qua)", 0, 30, 0)
        phys_hlth = st.slider("Số ngày sức khỏe thể chất không tốt (30 ngày qua)", 0, 30, 0)
    with p2:
        st.markdown("**Lối sống & thông tin cá nhân**")
        smoker = int(st.toggle("Hút thuốc"))
        stroke = int(st.toggle("Tiền sử đột quỵ"))
        phys_activity = int(st.toggle("Hoạt động thể chất"))
        fruits = int(st.toggle("Ăn trái cây thường xuyên"))
        veggies = int(st.toggle("Ăn rau thường xuyên"))
        alcohol = int(st.toggle("Uống rượu bia nhiều"))
        diff_walk = int(st.toggle("Khó khăn khi đi lại"))
        sex = int(st.toggle("Nam", help="Tắt: Nữ · Bật: Nam"))
        education = st.selectbox("Trình độ học vấn", list(EDU_LBL), index=5, format_func=lambda x: EDU_LBL[x])
        income = st.selectbox("Mức thu nhập", list(INC_LBL), index=6, format_func=lambda x: INC_LBL[x])

    thr_pct = st.slider("🎚️ Ngưỡng phân loại (%)", 10, 90, int(round(THR * 100)), 1, format="%d%%",
                        help=f"Ngưỡng mặc định do mô hình lựa chọn là {THR:.0%}.")
    threshold_input = thr_pct / 100
    st.caption(f"⭐ Ngưỡng khuyến nghị của mô hình: **{THR:.0%}** · Ngưỡng đang chọn: **{threshold_input:.0%}**")

    if st.button("🔍 Dự đoán nguy cơ", type="primary", width="stretch"):
        row = {"HighBP": high_bp, "HighChol": high_chol, "CholCheck": 1, "BMI": bmi, "Smoker": smoker,
               "Stroke": stroke, "Diabetes": diabetes, "PhysActivity": phys_activity, "Fruits": fruits,
               "Veggies": veggies, "HvyAlcoholConsump": alcohol, "AnyHealthcare": 1, "NoDocbcCost": 0,
               "GenHlth": gen_hlth, "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": diff_walk,
               "Sex": sex, "Age": age, "Education": education, "Income": income}
        x_in = pd.DataFrame([row])[features]
        risk_probability = float(model.predict_proba(scaler.transform(x_in))[0, 1])
        prediction = int(risk_probability >= threshold_input)

        st.divider()
        risk_percent = risk_probability * 100
        sections = [("#22c55e", min(max(risk_percent, 0), 25)),
                    ("#eab308", min(max(risk_percent - 25, 0), 25)),
                    ("#f97316", min(max(risk_percent - 50, 0), 25)),
                    ("#ef4444", min(max(risk_percent - 75, 0), 25))]
        risk_bar = "".join(
            f'<div style="flex:1;background:#374151;position:relative;">'
            f'<div style="width:{filled / 25 * 100:.2f}%;height:100%;background:{color};"></div></div>'
            for color, filled in sections)
        st.markdown(
            f"""
            <div role="img" aria-label="Xác suất ước tính theo mô hình {risk_percent:.1f}%">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem;">
                    <span style="font-weight:600;">Xác suất ước tính theo mô hình</span>
                    <span style="font-size:1.5rem;font-weight:700;">{risk_percent:.1f}%</span>
                </div>
                <div style="display:flex;height:18px;overflow:hidden;border-radius:9px;gap:2px;">{risk_bar}</div>
                <div style="display:flex;justify-content:space-between;color:#9ca3af;font-size:0.8rem;margin-top:0.3rem;">
                    <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True)
        if prediction == 1:
            st.error(f"⚠️ Mô hình ước tính nguy cơ ở mức cao — xác suất {risk_probability:.1%} "
                     f"≥ ngưỡng {threshold_input:.0%}")
        else:
            st.success(f"✅ Mô hình ước tính nguy cơ ở mức thấp — xác suất {risk_probability:.1%} "
                       f"< ngưỡng {threshold_input:.0%}")
        st.warning("Kết quả chỉ mang tính chất sàng lọc và hỗ trợ tham khảo, "
                   "không thay thế cho chẩn đoán hoặc tư vấn của bác sĩ.")

# ==============================================================================
# CHÂN TRANG
# ==============================================================================
st.divider()
f1, f2, f3, f4 = st.columns(4)
f1.caption(f"📁 Dữ liệu: CDC BRFSS ({len(data):,} hồ sơ)")
f2.caption(f"🧠 Thuật toán: {meta.get('model_name', 'Logistic Regression')}")
f3.caption(f"🎯 Chỉ số Test: ROC-AUC {M['ROC-AUC']:.3f} | PR-AUC {M['PR-AUC']:.3f}")
f4.caption(f"⚖️ Ngưỡng tối ưu: {THR * 100:.0f}%")
