import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# ==============================================================================
# CONFIG & PATHS
# ==============================================================================
BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "final" / "heart_disease_health_indicators_BRFSS2015_V2.csv"
MODEL = BASE / "src" / "model" / "Logisic Regression" / "model_detail" / "logistic_regression_final.pkl"
META = BASE / "src" / "model" / "Logisic Regression" / "model_detail" / "model_metadata.json"
TARGET = "HeartDiseaseorAttack"

st.set_page_config(
    page_title="CardioView - Bảng Điều Khiển Nguy Cơ Tim Mạch",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Medical Theme)
st.markdown(
    """
    <style>
    .block-container { max-width: 1550px; padding-top: 2rem; padding-bottom: 3rem; }
    .eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #ef4444;
        margin-bottom: 0.2rem;
    }
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f3f4f6;
        margin-bottom: 0.2rem;
    }
    .sub {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    [data-testid=stMetric] {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .section-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# LABELS & DICTIONARIES
# ==============================================================================
AGE_ORDER = [
    "18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
    "55–59", "60–64", "65–69", "70–74", "75–79", "80+"
]
AGE_LBL = dict(enumerate(AGE_ORDER, 1))
AGE_BINS = {
    "18–24": [1], "25–34": [2, 3], "35–44": [4, 5],
    "45–54": [6, 7], "55–64": [8, 9], "65–74": [10, 11], "75+": [12, 13]
}

INC_LBL = {
    1: "<$10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
    5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: "$75k+"
}
INC_BINS = {
    "< $25k": [1, 2, 3, 4],
    "$25–50k": [5, 6],
    "$50–75k": [7],
    "$75k+": [8]
}

EDU_LBL = {
    1: "Chưa đi học", 2: "Tiểu học", 3: "THPT dở dang",
    4: "Tốt nghiệp THPT", 5: "Đại học dở dang", 6: "Tốt nghiệp đại học"
}
EDU_BINS = {
    "THPT trở xuống": [1, 2, 3, 4],
    "Đại học dở dang": [5],
    "Tốt nghiệp đại học": [6]
}

DIA_LBL = {0: "Không", 1: "Tiền tiểu đường", 2: "Có"}
GEN_LBL = {1: "Xuất sắc", 2: "Rất tốt", 3: "Tốt", 4: "Trung bình", 5: "Kém"}
SEX_ORDER = ["Nữ", "Nam"]

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


# ==============================================================================
# DATA LOADING & MODEL CACHING
# ==============================================================================
@st.cache_data
def load():
    d = pd.read_csv(CSV)
    # Fix 3: Ép kiểu HeartDiseaseorAttack sang integer chuẩn xác để tránh lỗi .0 ca
    d[TARGET] = d[TARGET].fillna(0).astype(int)
    # Cân chỉnh mã giới tính (BRFSS: 1=Nam, 2=Nữ; chuẩn mô hình: 1=Nam, 0=Nữ)
    if 2.0 in d.Sex.values or 2 in d.Sex.values:
        d["Sex"] = d["Sex"].map({2.0: 0, 1.0: 1, 2: 0, 1: 1}).fillna(0).astype(int)
    # Tiền xử lý BMI hợp lệ từ 12 đến 60
    d = d[d.BMI.isna() | d.BMI.between(12, 60)].reset_index(drop=True)
    d["Age_lbl"] = pd.Categorical(d.Age.map(AGE_LBL), AGE_ORDER, ordered=True)
    d["Sex_lbl"] = d.Sex.map({0: "Nữ", 1: "Nam"})
    d["Inc_lbl"] = pd.Categorical(d.Income.map(INC_LBL), list(INC_LBL.values()), ordered=True)
    d["Edu_lbl"] = pd.Categorical(d.Education.map(EDU_LBL), list(EDU_LBL.values()), ordered=True)
    d["Bệnh tim"] = d[TARGET].map({0: "Không bệnh tim", 1: "Bệnh tim"})
    return d


@st.cache_resource
def get_model():
    model_data = joblib.load(MODEL)

    with open(META, encoding="utf-8") as f:
        meta = json.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]
    threshold = model_data["threshold"]

    return model, scaler, features, threshold, meta

@st.cache_data
def predict_all():
    model, scaler, features, _, _ = get_model()
    X = load()[features]
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]

@st.cache_data
def get_test_split_indices():
    # Fix 7: Tái tạo chính xác tập Test độc lập (test_size=0.2, random_state=42, stratify=y)
    d = load()
    _, test_idx = train_test_split(d.index, test_size=0.2, random_state=42, stratify=d[TARGET])
    return set(test_idx)


data = load()
model, scaler, features, THR, meta = get_model()
proba = pd.Series(predict_all(), index=data.index)
test_indices = get_test_split_indices()

# National baseline benchmark
NATIONAL_RATE = data[TARGET].mean() * 100

# ==============================================================================
# SIDEBAR FILTERS (MULTI-LEVEL FILTERING)
# ==============================================================================
DEFAULTS = {
    "f_state": "Tất cả các bang",
    "f_sex": "Tất cả giới tính",
    "f_age": "Tất cả nhóm tuổi",
    "f_bp": "Tất cả",
    "f_chol": "Tất cả",
    "f_dia": "Tất cả",
    "f_smk": "Tất cả",
    "f_act": "Tất cả",
    "f_inc": "Tất cả mức thu nhập",
    "f_edu": "Tất cả trình độ",
    "f_bmi": (12.0, 60.0),
    "f_eval_set": "Toàn bộ dữ liệu dân số",
    "ver": 0
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


with st.sidebar:
    st.markdown("## ❤️ CardioView")
    st.caption("Hệ thống Giám sát & Dự đoán Sức khỏe Dân số")
    st.divider()

    st.markdown("### 🗺️ Cấp 1: Phân bố Địa lý")
    state_list = ["Tất cả các bang"] + sorted([s for s in data["StateName"].dropna().unique() if str(s) != "nan"])
    st.selectbox("Bang / Vùng lãnh thổ", state_list, key="f_state")

    st.markdown("### 👤 Cấp 2: Nhân khẩu học")
    st.selectbox("Giới tính", ["Tất cả giới tính", "Nữ", "Nam"], key="f_sex")
    st.selectbox("Nhóm tuổi", ["Tất cả nhóm tuổi"] + list(AGE_BINS), key="f_age")
    st.selectbox("Thu nhập bình quân", ["Tất cả mức thu nhập"] + list(INC_BINS), key="f_inc")
    st.selectbox("Trình độ học vấn", ["Tất cả trình độ"] + list(EDU_BINS), key="f_edu")

    st.markdown("### 🩺 Cấp 3: Tiền sử Lâm sàng")
    st.selectbox("Huyết áp cao", ["Tất cả", "Không", "Có"], key="f_bp")
    st.selectbox("Cholesterol cao", ["Tất cả", "Không", "Có"], key="f_chol")
    st.selectbox("Tiểu đường", ["Tất cả", "Không", "Tiền tiểu đường", "Có"], key="f_dia")

    st.markdown("### 🏃 Cấp 4: Lối sống & Thể trạng")
    st.selectbox("Hút thuốc", ["Tất cả", "Không", "Có"], key="f_smk")
    st.selectbox("Vận động thể chất", ["Tất cả", "Không", "Có"], key="f_act")
    st.slider("Chỉ số khối cơ thể (BMI)", 12.0, 60.0, step=0.5, key="f_bmi",
              help="Dữ liệu đã được làm sạch trong khoảng sinh học hợp lý 12 - 60 kg/m²")

    st.markdown("### 🧪 Phân chia Dữ liệu Đánh giá")
    st.radio(
        "Tập dữ liệu hiển thị:",
        ["Toàn bộ dữ liệu dân số", "Chỉ tập Test độc lập (50,736 mẫu)"],
        key="f_eval_set",
        help="Cho phép đánh giá mô hình chuẩn xác trên tập Test độc lập khớp với metadata."
    )

    st.divider()
    st.button("↻ Đặt lại tất cả bộ lọc", use_container_width=True, on_click=reset)

    st.caption("TIẾN TRÌNH ĐỘ PHỦ DỮ LIỆU")


def yn(d, col, v):
    return d if v == "Tất cả" else d[d[col] == (1 if v == "Có" else 0)]


S = st.session_state

# Fix 7: Lọc theo tập Test nếu người dùng chọn
base_data = data
if S.f_eval_set == "Chỉ tập Test độc lập (50,736 mẫu)":
    base_data = base_data[base_data.index.isin(test_indices)]

# Fix 4: base_geo giữ nguyên TOÀN BỘ CÁC BANG (chưa lọc theo bang) để bản đồ và Top 5 không bị mất ngữ cảnh
base_geo = base_data
if S.f_sex != "Tất cả giới tính":
    base_geo = base_geo[base_geo.Sex_lbl == S.f_sex]
if S.f_age != "Tất cả nhóm tuổi":
    base_geo = base_geo[base_geo.Age.isin(AGE_BINS[S.f_age])]
if S.f_inc != "Tất cả mức thu nhập":
    base_geo = base_geo[base_geo.Income.isin(INC_BINS[S.f_inc])]
if S.f_edu != "Tất cả trình độ":
    base_geo = base_geo[base_geo.Education.isin(EDU_BINS[S.f_edu])]
if S.f_dia != "Tất cả":
    base_geo = base_geo[base_geo.Diabetes == {"Không": 0, "Tiền tiểu đường": 1, "Có": 2}[S.f_dia]]

base_geo = yn(yn(yn(yn(base_geo, "HighBP", S.f_bp), "HighChol", S.f_chol), "Smoker", S.f_smk), "PhysActivity", S.f_act)
base_geo = base_geo[base_geo.BMI.isna() | base_geo.BMI.between(*S.f_bmi)]

# Áp dụng bộ lọc bang cho tập dữ liệu chi tiết
base = base_geo
if S.f_state != "Tất cả các bang":
    base = base[base.StateName == S.f_state]

with st.sidebar:
    cov_ratio = len(base) / len(base_data)
    st.progress(cov_ratio)
    st.caption(f"**{len(base):,} / {len(base_data):,}** bản ghi phù hợp ({cov_ratio * 100:.1f}% tập dữ liệu).")


# ==============================================================================
# CROSS-FILTERING (INTER-CHART LINKAGE)
# ==============================================================================
def picked(key, order):
    ev = st.session_state.get(key)
    if not ev or "selection" not in ev or not ev["selection"].get("points"):
        return []
    return [order[p["point_number"]] for p in ev["selection"]["points"] if
            p.get("point_number") is not None and p["point_number"] < len(order)]


AGE_KEY = f"age_{S.ver}"
SEX_KEY = f"sex_{S.ver}"
MAP_KEY = f"map_{S.ver}"

age_sel = picked(AGE_KEY, AGE_ORDER)
sex_sel = picked(SEX_KEY, SEX_ORDER)

# Fix 5: Bắt sự kiện nhấp chuột trực tiếp trên Bản đồ để chọn bang
map_ev = st.session_state.get(MAP_KEY)
if map_ev and "selection" in map_ev and map_ev["selection"].get("points"):
    clicked_pts = map_ev["selection"]["points"]
    if clicked_pts:
        clicked_loc = clicked_pts[0].get("location")
        if clicked_loc and clicked_loc in CODE_TO_STATE:
            clicked_state_name = CODE_TO_STATE[clicked_loc]
            if S.f_state != clicked_state_name:
                st.session_state.f_state = clicked_state_name
                st.rerun()


def by_age(d):
    return d[d.Age_lbl.isin(age_sel)] if age_sel else d


def by_sex(d):
    return d[d.Sex_lbl.isin(sex_sel)] if sex_sel else d


# Final filtered slice with cross-filter applied
df = by_sex(by_age(base))


# Chart styling helper
def sty(fig, h=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=h,
        margin=dict(l=15, r=15, t=35, b=15),
        font=dict(family="sans-serif", size=12),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center")
    )
    return fig


def rate_yes_no(d, cols):
    rows = []
    for lbl, s in cols.items():
        for v, name in [(1, "Có"), (0, "Không")]:
            m = d[TARGET][s.reindex(d.index) == v]
            rows.append((lbl, name, m.mean() * 100 if len(m) else np.nan))
    return pd.DataFrame(rows, columns=["Yếu tố", "Tình trạng", "Tỷ lệ bệnh tim (%)"])


def yn_bar(d, cols, title):
    fig = px.bar(
        rate_yes_no(d, cols),
        x="Yếu tố",
        y="Tỷ lệ bệnh tim (%)",
        color="Tình trạng",
        barmode="group",
        title=title,
        color_discrete_map={"Có": "#ef4444", "Không": "#38bdf8"},
        text_auto=".1f"
    )
    fig.update_layout(legend_title_text="Tình trạng")
    return sty(fig, 320)


# ==============================================================================
# HEADER & EXECUTIVE KPI SUMMARY
# ==============================================================================
st.markdown('<div class="eyebrow">HỆ THỐNG TRỰC QUAN HÓA THÔNG TIN Y TẾ & SỨC KHỎE DÂN SỐ</div>',
            unsafe_allow_html=True)
st.markdown('<div class="page-title">Bảng Điều Khiển Phân Tích & Dự Đoán Nguy Cơ Tim Mạch</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Khám phá toàn diện phân bố không gian địa lý, tương quan nhân khẩu học, '
    'bệnh lý lâm sàng, lối sống và phân tầng nguy cơ tim mạch dựa trên Machine Learning. '
    'Hỗ trợ lọc đa cấp, Drill-down Treemap và Cross-filtering (nhấp vào cột/lát cắt/bản đồ để lọc chéo).</div>',
    unsafe_allow_html=True
)

if len(df) == 0:
    st.warning("⚠️ Không có bản ghi nào phù hợp với điều kiện bộ lọc hiện tại. Vui lòng nới lỏng tiêu chí lọc.")
    st.stop()

# Hiển thị thông báo trạng thái lọc chéo
active_cross = []
if age_sel:
    active_cross.append(f"Nhóm tuổi: {', '.join(age_sel)}")
if sex_sel:
    active_cross.append(f"Giới tính: {', '.join(sex_sel)}")

if active_cross or S.f_state != "Tất cả các bang":
    c1, c2, c3 = st.columns([4, 1, 1])
    status_text = []
    if S.f_state != "Tất cả các bang":
        status_text.append(f"📍 Bang: **{S.f_state}**")
    if active_cross:
        status_text.append(f"🔍 Lọc chéo: **{' | '.join(active_cross)}**")
    c1.info(" · ".join(status_text))
    if active_cross:
        c2.button("✕ Bỏ lọc chéo", on_click=clear_cross_selection, use_container_width=True)
    if S.f_state != "Tất cả các bang":
        c3.button("🌐 Xem toàn quốc", on_click=reset_state_filter, use_container_width=True)

# Executive KPI Metrics
k = st.columns(5)
curr_rate = df[TARGET].mean() * 100
delta_nat = curr_rate - NATIONAL_RATE

with k[0].container(height=165):
    st.metric("Tổng số bản ghi", f"{len(df):,}", help="Số người tham gia khảo sát phù hợp bộ lọc")
with k[1].container(height=165):
    st.metric(
        "Tỷ lệ bệnh tim mạch",
        f"{curr_rate:.1f}%",
        delta=f"{delta_nat:+.1f}% so với toàn quốc",
        delta_color="inverse",
        help="Tỷ lệ người mắc bệnh tim hoặc từng đau tim trong mẫu"
    )
with k[2].container(height=165):
    st.metric("Nhóm tuổi chiếm ưu thế", str(df.Age_lbl.value_counts().idxmax()),
              help="Nhóm độ tuổi có số lượng người đông nhất")
with k[3].container(height=165):
    st.metric("Chỉ số BMI trung bình", f"{df.BMI.mean():.1f}", help="Chỉ số khối cơ thể trung bình của nhóm")
with k[4].container(height=165):
    st.metric("Tỷ lệ tăng huyết áp", f"{df.HighBP.mean() * 100:.1f}%", help="Tỷ lệ người có tiền sử huyết áp cao")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: GEOGRAPHIC SPATIAL DISTRIBUTION (MAP REQUIREMENT)
# ==============================================================================
with st.container(border=True):
    st.markdown("### 1. 🗺️ Bản đồ Địa lý: Phân bố Không gian Nguy cơ Tim mạch `[Click để chọn bang]`")
    st.caption(
        "Bản đồ Choropleth thể hiện tỷ lệ mắc bệnh tim mạch trên toàn bộ 53 bang/vùng lãnh thổ của Hoa Kỳ. Bấm trực tiếp vào một bang trên bản đồ để lọc dữ liệu.")

    # Fix 4: state_agg tính từ base_geo (toàn bộ các bang) để giữ nguyên ngữ cảnh toàn quốc
    state_agg = base_geo.groupby("StateName", observed=True)[TARGET].agg(
        TongSo="count",
        SoCa="sum",
        TyLe="mean"
    ).reset_index()
    # Fix 3: Ép kiểu SoCa sang int để không bị hiển thị 1,234.0 ca
    state_agg["SoCa"] = state_agg["SoCa"].astype(int)
    state_agg["TyLe_pct"] = state_agg["TyLe"] * 100
    state_agg["Code"] = state_agg["StateName"].map(STATE_CODE_MAP)

    map_col, drill_col = st.columns([3, 1])

    with map_col:
        # Chart 1: Choropleth Geographic Map
        fig_map = px.choropleth(
            state_agg,
            locations="Code",
            locationmode="USA-states",
            color="TyLe_pct",
            scope="usa",
            hover_name="StateName",
            hover_data={
                "Code": False,
                "TyLe_pct": ":.2f",
                "TongSo": ":,",
                "SoCa": ":,"
            },
            color_continuous_scale="Reds",
            labels={
                "TyLe_pct": "Tỷ lệ bệnh tim (%)",
                "TongSo": "Số mẫu khảo sát",
                "SoCa": "Số ca bệnh tim"
            }
        )
        fig_map.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(255,255,255,0.05)"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=430,
            coloraxis_colorbar=dict(title="Tỷ lệ (%)", thickness=14, len=0.7)
        )
        # Fix 5: on_select="rerun" cho phép click trực tiếp vào bang trên bản đồ
        st.plotly_chart(
            fig_map,
            key=MAP_KEY,
            on_select="rerun",
            selection_mode="points",
            use_container_width=True
        )

    with drill_col:
        st.markdown("##### 📍 Xếp hạng & So sánh Bang")

        # Fix 4: Luôn hiển thị Top 5 cao nhất và thấp nhất ngay cả khi đã chọn 1 bang
        rank_tab1, rank_tab2 = st.tabs(["🔴 Top 5 Cao nhất", "🟢 Top 5 Thấp nhất"])
        with rank_tab1:
            top_high = state_agg.sort_values(by="TyLe_pct", ascending=False).head(5)
            for _, r in top_high.iterrows():
                # Fix 3: int(r['SoCa']) hiển thị chuẩn dạng số nguyên
                st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({int(r['SoCa']):,} ca)")
        with rank_tab2:
            top_low = state_agg.sort_values(by="TyLe_pct", ascending=True).head(5)
            for _, r in top_low.iterrows():
                st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({int(r['SoCa']):,} ca)")

        if S.f_state != "Tất cả các bang":
            st.divider()
            st_data = state_agg[state_agg.StateName == S.f_state]
            if len(st_data) > 0:
                st_r = st_data.iloc[0]
                st.markdown(f"**Bang đang chọn: {S.f_state}**")
                st.metric("Tỷ lệ tại bang", f"{st_r['TyLe_pct']:.1f}%",
                          delta=f"{st_r['TyLe_pct'] - NATIONAL_RATE:+.1f}% vs Toàn quốc", delta_color="inverse")
                # Fix 3: int(st_r['SoCa'])
                st.caption(f"Tổng: {int(st_r['SoCa']):,} ca / {int(st_r['TongSo']):,} bản ghi")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: DEMOGRAPHICS & CROSS-FILTERING (CHARTS 2 & 3)
# ==============================================================================
st.markdown("### 2. 👥 Nhân khẩu học & Liên kết Lọc chéo (Cross-filtering)")
st.caption(
    "Nhấp chuột vào một cột Nhóm tuổi hoặc lát cắt Giới tính để kích hoạt liên kết lọc chéo dữ liệu giữa các biểu đồ.")

r1a, r1b = st.columns(2)

with r1a.container(border=True):
    # Chart 2: Bar Chart with Cross-filter
    st.markdown("**Biểu đồ Cột: Tỷ lệ bệnh tim theo Nhóm tuổi** `[Cross-filter]`")
    st.caption("Bấm chọn vào cột để lọc chéo theo nhóm tuổi")
    d_age = by_sex(base)
    rate_age = d_age.groupby("Age_lbl", observed=False)[TARGET].mean().reindex(AGE_ORDER) * 100
    colors_age = ["#ef4444" if (not age_sel or a_ in age_sel) else "#4b5563" for a_ in AGE_ORDER]
    fig_age = go.Figure(
        go.Bar(
            x=AGE_ORDER,
            y=rate_age.values,
            marker_color=colors_age,
            name="Tỷ lệ bệnh tim (%)",
            text=[f"{v:.1f}%" for v in rate_age.values],
            textposition="outside",
            hovertemplate="<b>Nhóm tuổi %{x}</b><br>Tỷ lệ bệnh tim: %{y:.2f}%<extra></extra>"
        )
    )
    fig_age.update_layout(yaxis_title="Tỷ lệ bệnh tim (%)", xaxis_title="Nhóm tuổi", showlegend=False)
    st.plotly_chart(
        sty(fig_age, 340),
        key=AGE_KEY,
        on_select="rerun",
        selection_mode="points",
        use_container_width=True
    )

with r1b.container(border=True):
    # Chart 3: Donut / Pie Chart with Cross-filter
    # Fix 6: Ghi rõ đây là Cơ cấu ca bệnh (tỷ trọng %), đồng thời bổ sung Tỷ lệ mắc bệnh trong từng giới
    st.markdown("**Biểu đồ Tròn: Cơ cấu ca bệnh theo Giới tính** `[Cross-filter]`")
    d_sex = by_age(base)
    cases_sex = d_sex[d_sex[TARGET] == 1].Sex_lbl.value_counts().reindex(SEX_ORDER).fillna(0).astype(int)

    # Tính tỷ lệ mắc (Prevalence) của từng giới để phân biệt rõ với tỷ trọng ca bệnh
    fem_sub = d_sex[d_sex.Sex_lbl == "Nữ"]
    male_sub = d_sex[d_sex.Sex_lbl == "Nam"]
    fem_prev = (fem_sub[TARGET].mean() * 100) if len(fem_sub) else 0.0
    male_prev = (male_sub[TARGET].mean() * 100) if len(male_sub) else 0.0
    st.caption(
        f"Tỷ lệ mắc trong từng giới: Nam **{male_prev:.1f}%** | Nữ **{fem_prev:.1f}%** (Bấm lát cắt để lọc chéo)")

    fig_sex = go.Figure(
        go.Pie(
            labels=SEX_ORDER,
            values=cases_sex.values,
            hole=0.58,
            sort=False,
            marker=dict(colors=["#f472b6", "#38bdf8"]),
            textinfo="label+percent",
            hovertemplate="<b>Giới tính: %{label}</b><br>Số ca bệnh tim: %{value:,}<br>Tỷ trọng ca bệnh: %{percent}<extra></extra>"
        )
    )
    fig_sex.update_layout(legend_title_text="Giới tính")
    st.plotly_chart(
        sty(fig_sex, 340),
        key=SEX_KEY,
        on_select="rerun",
        selection_mode="points",
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: CLINICAL FACTORS, TRENDS & BMI (CHARTS 4, 5, 6)
# ==============================================================================
st.markdown("### 3. 🩺 Yếu tố Lâm sàng & Phân tích Sinh trắc học")
r2a, r2b = st.columns(2)

with r2a.container(border=True):
    # Chart 4: Line Chart with Markers
    st.markdown("**Biểu đồ Đường: Xu hướng nguy cơ theo Độ tuổi × Tiền sử Cao huyết áp**")
    st.caption("So sánh tốc độ gia tăng nguy cơ tim mạch theo tuổi thọ giữa người huyết áp cao và bình thường.")
    g_line = (
        df.dropna(subset=["HighBP"])
        .assign(BP=lambda x: x.HighBP.map({0: "Bình thường",1: "Cao"}))
        .groupby(["Age_lbl", "BP"], observed=True)[TARGET]
        .mean()
        .mul(100)
        .reset_index(name="Tỷ lệ bệnh tim (%)")
    )
    fig_line = px.line(
        g_line,
        x="Age_lbl",
        y="Tỷ lệ bệnh tim (%)",
        color="BP",
        markers=True,
        labels={"Age_lbl": "Nhóm tuổi"},
        color_discrete_map={"Huyết áp cao": "#ef4444", "Huyết áp bình thường": "#38bdf8"}
    )
    fig_line.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b> (%{data.name})<br>Tỷ lệ: %{y:.2f}%<extra></extra>"
    )
    fig_line.update_layout(
        legend_title_text="Tình trạng Huyết áp",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )    
    st.plotly_chart(sty(fig_line, 330), use_container_width=True)

with r2b.container(border=True):
    # Charts 5 & 6: Histogram & Box Plot with Tabs
    # Fix 8: Chuẩn hóa nhãn Tabs là "Chuyển đổi góc nhìn trực quan", giải thích đúng khoảng BMI 12 - 60
    st.markdown("**Phân tích Thể trạng: Chỉ số BMI theo Tình trạng Bệnh tim**")
    tab_dist, tab_box = st.tabs(["📊 Phân bố tần suất (Histogram)", "📦 Biểu đồ Hộp & Điểm phân vị (Box Plot)"])
    dd_bmi = df.dropna(subset=["BMI"])

    with tab_dist:
        # Chart 5: Histogram
        fig_hist = px.histogram(
            dd_bmi,
            x="BMI",
            color="Bệnh tim",
            nbins=45,
            barmode="overlay",
            histnorm="percent",
            opacity=0.68,
            color_discrete_map={"Bệnh tim": "#ef4444", "Không bệnh tim": "#38bdf8"}
        )
        fig_hist.update_layout(
            legend_title_text="Nhóm",
            yaxis_title="% Tỷ trọng trong nhóm",
            xaxis_title="Chỉ số khối cơ thể BMI (kg/m²)"
        )
        st.plotly_chart(sty(fig_hist, 290), use_container_width=True)

    with tab_box:
        # Chart 6: Box Plot
        fig_box = px.box(
            dd_bmi,
            x="Bệnh tim",
            y="BMI",
            color="Bệnh tim",
            points="outliers",
            color_discrete_map={"Bệnh tim": "#ef4444", "Không bệnh tim": "#38bdf8"}
        )
        fig_box.update_layout(
            legend_title_text="Nhóm",
            xaxis_title="Tình trạng bệnh tim",
            yaxis_title="BMI (kg/m²) [Giới hạn hợp lệ 12–60]"
        )
        st.plotly_chart(sty(fig_box, 290), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: MULTI-FACTOR COMPARISON & SCATTER INTERACTION (CHARTS 7 & 8)
# ==============================================================================
st.markdown("### 4. 🧬 Gánh nặng Bệnh lý nền & Tương quan Thể trạng")
r3a, r3b = st.columns(2)

with r3a.container(border=True):
    st.markdown("**Gánh nặng Đa bệnh lý & Tương quan Thể trạng**")
    tab_dis, tab_scat = st.tabs(["📊 Các bệnh lý mạn tính liên quan", "🫧 Tương quan Bong bóng (Scatter/Bubble)"])

    with tab_dis:
        conds = {
            "Đột quỵ": df.Stroke,
            "Cholesterol cao": df.HighChol,
            "Huyết áp cao": df.HighBP,
            "Khó đi lại": df.DiffWalk,
            "Tiểu đường": df.Diabetes.map({0: 0, 1: 0, 2: 1})
        }
        st.plotly_chart(yn_bar(df, conds, "Tỷ lệ bệnh tim theo từng bệnh lý nền kèm theo"), use_container_width=True)

    with tab_scat:
        # Chart 7: Scatter / Bubble Chart
        s_bubble = (
            df.dropna(subset=["GenHlth", "BMI"])
            .assign(GH=lambda x: x.GenHlth.map(GEN_LBL))
            .groupby(["GH", "Age_lbl"], observed=True)
            .agg(BMI=("BMI", "mean"), Rate=(TARGET, "mean"), Records=(TARGET, "size"))
            .reset_index()
        )
        s_bubble["Rate"] *= 100
        fig_scat = px.scatter(
            s_bubble,
            x="BMI",
            y="Rate",
            size="Records",
            color="GH",
            hover_name="Age_lbl",
            labels={
                "Rate": "Tỷ lệ bệnh tim (%)",
                "GH": "Sức khỏe tự đánh giá",
                "Records": "Số bản ghi",
                "BMI": "BMI trung bình"
            },
            category_orders={"GH": list(GEN_LBL.values())},
            title="Tương quan BMI - Tỷ lệ bệnh tim theo Nhóm Sức khỏe"
        )
        st.plotly_chart(sty(fig_scat, 320), use_container_width=True)

with r3b.container(border=True):
    st.markdown("**Yếu tố Thói quen Lối sống**")
    life_factors = {
        "Hút thuốc": df.Smoker,
        "Vận động thể chất": df.PhysActivity,
        "Ăn trái cây": df.Fruits,
        "Ăn rau củ": df.Veggies,
        "Uống rượu nhiều": df.HvyAlcoholConsump
    }
    st.plotly_chart(yn_bar(df, life_factors, "Tỷ lệ bệnh tim giữa nhóm Có vs Không theo lối sống"),
                    use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: SOCIOECONOMIC TREEMAP & RISK INTERACTION HEATMAP (CHARTS 8, 9, 10)
# ==============================================================================
st.markdown("### 5. 💰 Kinh tế - Xã hội & Ma trận Tương tác Nguy cơ")
r4a, r4b = st.columns(2)

with r4a.container(border=True):
    # Chart 8: Treemap with Hierarchical Drill-down (Fix 8: Đúng nghĩa Drill-down phân cấp)
    st.markdown("**Biểu đồ Cây Phân cấp: Thu nhập × Học vấn (Treemap)** `[Drill-down Phân cấp]`")
    st.caption(
        "Nhấp chuột vào từng khối Thu nhập để đào sâu (Drill-down) vào các bậc Học vấn bên trong. Kích thước = Số bản ghi, Màu = Tỷ lệ bệnh tim.")
    tm_df = (
        df.groupby(["Inc_lbl", "Edu_lbl"], observed=True)
        .agg(Records=(TARGET, "size"), Rate=(TARGET, "mean"))
        .reset_index()
    )
    tm_df["Rate"] *= 100
    fig_tree = px.treemap(
        tm_df,
        path=[px.Constant("Tất cả mức thu nhập"), "Inc_lbl", "Edu_lbl"],
        values="Records",
        color="Rate",
        color_continuous_scale="RdYlGn_r",
        labels={
            "Rate": "Tỷ lệ bệnh tim (%)",
            "Records": "Số bản ghi",
            "Inc_lbl": "Mức thu nhập",
            "Edu_lbl": "Trình độ học vấn"
        }
    )
    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Số bản ghi: %{value:,}<br>Tỷ lệ bệnh tim: %{color:.2f}%<extra></extra>"
    )
    st.plotly_chart(sty(fig_tree, 340), use_container_width=True)

with r4b.container(border=True):
    # Chart 9: Heatmap
    st.markdown("**Bản đồ Nhiệt Ma trận: Tương tác Huyết áp cao × Cholesterol cao**")
    st.caption("Đánh giá hiệu ứng cộng gộp nguy cơ tim mạch khi mắc đồng thời 2 yếu tố nguy cơ tim mạch hàng đầu.")
    h_data = df.dropna(subset=["HighBP", "HighChol"])
    piv = (
        h_data.groupby(["HighBP", "HighChol"])[TARGET]
        .mean()
        .mul(100)
        .unstack()
        .rename(
            index={0: "Huyết áp bình thường", 1: "Huyết áp cao"},
            columns={0: "Cholesterol bình thường", 1: "Cholesterol cao"}
        )
    )
    fig_heat = px.imshow(
        piv,
        text_auto=".1f",
        color_continuous_scale="Reds",
        aspect="auto",
        labels={"color": "Tỷ lệ bệnh tim (%)"}
    )
    fig_heat.update_traces(
        hovertemplate="Huyết áp: %{y}<br>Cholesterol: %{x}<br>Tỷ lệ bệnh tim: %{z:.2f}%<extra></extra>"
    )
    st.plotly_chart(sty(fig_heat, 340), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 6: MINI MODEL PERFORMANCE DASHBOARD
# ==============================================================================

st.markdown("### 6. 🧠 Đánh giá Mô hình Logistic Regression")
st.caption(
    "Kết quả dưới đây được tính trên tập Test độc lập, "
    "nhằm đánh giá khả năng phân tầng nguy cơ của mô hình."
)

# --------------------------------------------------------------------------
# Prepare independent Test set
# --------------------------------------------------------------------------

test_df = data.loc[list(test_indices)].copy()
y_test = test_df[TARGET]

# Xác suất dự đoán tương ứng với đúng index của Test set
p_test = proba.loc[test_df.index]

# Prediction theo ngưỡng sàng lọc hiện tại
y_pred_test = (p_test >= THR).astype(int)

# --------------------------------------------------------------------------
# KPI
# --------------------------------------------------------------------------

m = meta["evaluation"]

mk1, mk2, mk3, mk4 = st.columns(4)

with mk1:
    st.metric(
        "Accuracy",
        f"{m['accuracy']:.1%}",
        help="Tỷ lệ dự đoán đúng trên tập Test."
    )

with mk2:
    st.metric(
        "Precision",
        f"{m['precision']:.1%}",
        help="Trong các trường hợp được dự đoán là có bệnh tim, tỷ lệ dự đoán đúng."
    )

with mk3:
    st.metric(
        "F1-score",
        f"{m['f1_score']:.1%}",
        help="Chỉ số cân bằng giữa Precision và Recall."
    )

with mk4:
    st.metric(
        "Recall",
        f"{m['recall']:.1%}",
        help="Trong các trường hợp thực sự có bệnh tim, tỷ lệ được mô hình phát hiện."
    )

# --------------------------------------------------------------------------
# Probability Distribution
# --------------------------------------------------------------------------

c1, c2 = st.columns(2)

with c1.container(border=True):

    st.markdown("**Histogram: Phân phối xác suất dự đoán**")

    hist_df = pd.DataFrame({
        "Xác suất dự đoán (%)": p_test.values * 100,
        "Tình trạng thực tế": y_test.map({
            0: "Không bệnh tim",
            1: "Bệnh tim"
        }).values
    })

    fig_hist_model = px.histogram(
        hist_df,
        x="Xác suất dự đoán (%)",
        color="Tình trạng thực tế",
        nbins=40,
        barmode="overlay",
        opacity=0.65,
        histnorm="percent",
        color_discrete_map={
            "Không bệnh tim": "#38bdf8",
            "Bệnh tim": "#ef4444"
        },
        labels={
            "Xác suất dự đoán (%)": "Xác suất nguy cơ dự đoán (%)",
            "Tình trạng thực tế": "Thực tế"
        }
    )

    fig_hist_model.add_vline(
        x=THR * 100,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2.5,
        annotation_text=f"Ngưỡng {THR:.0%}",
        annotation_position="top right"
    )

    fig_hist_model.update_layout(
        yaxis_title="Tỷ trọng (%)",
        xaxis_title="Xác suất nguy cơ dự đoán (%)",
        legend_title="Tình trạng thực tế"
    )

    st.plotly_chart(
        sty(fig_hist_model, 330),
        use_container_width=True
    )


with c2.container(border=True):

    st.markdown("**Violin Plot: Phân bố điểm nguy cơ theo thực tế**")

    violin_df = pd.DataFrame({
        "Xác suất nguy cơ (%)": p_test.values * 100,
        "Tình trạng thực tế": y_test.map({
            0: "Không bệnh tim",
            1: "Bệnh tim"
        }).values
    })

    fig_violin = px.violin(
        violin_df,
        x="Tình trạng thực tế",
        y="Xác suất nguy cơ (%)",
        color="Tình trạng thực tế",
        box=True,
        points=False,
        color_discrete_map={
            "Không bệnh tim": "#38bdf8",
            "Bệnh tim": "#ef4444"
        },
        labels={
            "Tình trạng thực tế": "Tình trạng thực tế",
            "Xác suất nguy cơ (%)": "Xác suất nguy cơ dự đoán (%)"
        }
    )

    fig_violin.add_hline(
        y=THR * 100,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Ngưỡng {THR:.0%}",
        annotation_position="top right"
    )

    fig_violin.update_layout(
        showlegend=False,
        yaxis_title="Xác suất nguy cơ dự đoán (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        sty(fig_violin, 330),
        use_container_width=True
    )

# --------------------------------------------------------------------------
# Confusion Matrix + Feature Importance
# --------------------------------------------------------------------------

c3, c4 = st.columns(2)

with c3.container(border=True):

    st.markdown(
        f"**Confusion Matrix tại ngưỡng {THR:.0%}**"
    )

    cm = confusion_matrix(
        y_test,
        y_pred_test,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Thực tế: Không bệnh", "Thực tế: Bệnh"],
        columns=["Dự đoán: Không bệnh", "Dự đoán: Bệnh"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=",",
        color_continuous_scale="Reds",
        aspect="auto",
        labels={"color": "Số lượng"}
    )

    fig_cm.update_layout(
        xaxis_title="Dự đoán",
        yaxis_title="Thực tế"
    )

    st.plotly_chart(
        sty(fig_cm, 330),
        use_container_width=True
    )


with c4.container(border=True):

    st.markdown("**Hệ số Logistic Regression: Yếu tố ảnh hưởng đến mô hình**")

    importance_df = pd.DataFrame({
        "Yếu tố": features,
        "Coefficient": model.coef_[0]
    }).sort_values(
        "Coefficient",
        ascending=True
    ).tail(10)

    fig_imp = px.bar(
        importance_df,
        x="Coefficient",
        y="Yếu tố",
        orientation="h",
        text_auto=".2f",
        labels={
            "Coefficient": "Hệ số Logistic Regression",
            "Yếu tố": "Biến đầu vào"
        }
    )

    fig_imp.update_layout(
        yaxis_title="",
        xaxis_title="Hệ số Logistic Regression",
        showlegend=False
    )

    st.plotly_chart(
        sty(fig_imp, 330),
        use_container_width=True
    )
# ==============================================================================
# INDIVIDUAL RISK PREDICTION
# ==============================================================================

st.markdown("### 7. 🩺 Dự đoán nguy cơ cá nhân")

st.caption(
    "Nhập thông tin sức khỏe và lối sống để mô hình Logistic Regression "
    "ước tính nguy cơ mắc bệnh tim mạch."
)

with st.container(border=True):

    # --------------------------------------------------------------------------
    # INPUT
    # --------------------------------------------------------------------------

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**Tình trạng sức khỏe**")

        high_bp = int(st.toggle("Cao huyết áp"))

        high_chol = int(st.toggle("Cholesterol cao"))

        bmi = st.number_input(
            "BMI",
            min_value=12.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        diabetes = st.selectbox(
            "Tiểu đường",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Không",
                1: "Tiền tiểu đường",
                2: "Có"
            }[x]
        )

        gen_hlth = st.selectbox(
            "Tình trạng sức khỏe tổng quát",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Rất tốt",
                2: "Tốt",
                3: "Khá",
                4: "Kém",
                5: "Rất kém"
            }[x]
        )

        age = st.slider(
            "Nhóm tuổi",
            min_value=1,
            max_value=13,
            value=7
        )

        ment_hlth = st.slider(
            "Số ngày sức khỏe tinh thần không tốt",
            0,
            30,
            0
        )

        phys_hlth = st.slider(
            "Số ngày sức khỏe thể chất không tốt",
            0,
            30,
            0
        )

    with p2:
        st.markdown("**Lối sống & thông tin cá nhân**")

        smoker = int(st.toggle("Hút thuốc"))

        stroke = int(st.toggle("Tiền sử đột quỵ"))

        phys_activity = int(st.toggle("Hoạt động thể chất"))

        fruits = int(st.toggle("Ăn trái cây thường xuyên"))

        veggies = int(st.toggle("Ăn rau thường xuyên"))

        alcohol = int(st.toggle("Uống rượu bia nhiều"))

        diff_walk = int(st.toggle("Khó khăn khi đi lại"))

        sex = int(
            st.toggle(
                "Nam",
                help="Tắt: Nữ · Bật: Nam"
            )
        )

        education = st.selectbox(
            "Trình độ học vấn",
            list(range(1, 7))
        )

        income = st.selectbox(
            "Mức thu nhập",
            list(range(1, 9))
        )

    # --------------------------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------------------------

    threshold_input = st.slider(
        "🎚️ Ngưỡng phân loại",
        min_value=0.10,
        max_value=0.90,
        value=THR,
        step=0.01,
        format="%.0f%%",
        help="Ngưỡng mặc định được mô hình lựa chọn là 37%."
    )

    st.caption(
        f"⭐ Ngưỡng khuyến nghị của mô hình: **{THR:.0%}** · "
        f"Ngưỡng đang chọn: **{threshold_input:.0%}**"
    )

    # --------------------------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------------------------

    if st.button(
        "🔍 Dự đoán nguy cơ",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": 1,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "Diabetes": diabetes,
            "PhysActivity": phys_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": gen_hlth,
            "MentHlth": ment_hlth,
            "PhysHlth": phys_hlth,
            "DiffWalk": diff_walk,
            "Sex": sex,
            "Age": age,
            "Education": education,
            "Income": income
        }])

        # Đảm bảo đúng thứ tự feature của model
        input_data = input_data[features]

        # Scaling giống quá trình huấn luyện
        input_scaled = scaler.transform(input_data)

        # Xác suất thuộc nhóm có bệnh tim
        risk_probability = model.predict_proba(
            input_scaled
        )[0, 1]

        # Phân loại theo threshold người dùng đang chọn
        prediction = int(
            risk_probability >= threshold_input
        )

        st.divider()

        # ----------------------------------------------------------------------
        # RESULT
        # ----------------------------------------------------------------------

        risk_percent = risk_probability * 100
        risk_sections = [
            ("#22c55e", min(max(risk_percent, 0), 25)),
            ("#eab308", min(max(risk_percent - 25, 0), 25)),
            ("#f97316", min(max(risk_percent - 50, 0), 25)),
            ("#ef4444", min(max(risk_percent - 75, 0), 25)),
        ]
        risk_bar = "".join(
            f'<div style="flex:1;background:#374151;position:relative;">'
            f'<div style="width:{filled / 25:.6f}%;height:100%;background:{color};"></div>'
            f'</div>'
            for color, filled in risk_sections
        )
        st.markdown(
            f"""
            <div role="img" aria-label="Xác suất nguy cơ {risk_percent:.1f}%">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem;">
                    <span style="font-weight:600;">Xác suất nguy cơ</span>
                    <span style="font-size:1.5rem;font-weight:700;">{risk_percent:.1f}%</span>
                </div>
                <div style="display:flex;height:18px;overflow:hidden;border-radius:9px;gap:2px;">
                    {risk_bar}
                </div>
                <div style="display:flex;justify-content:space-between;color:#9ca3af;font-size:0.8rem;margin-top:0.3rem;">
                    <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error( f"⚠️ Nguy cơ cao — xác suất {risk_probability:.1%} " f"≥ ngưỡng {threshold_input:.0%}" )
        else:
            st.success( f"✅ Nguy cơ thấp — xác suất {risk_probability:.1%} " f"< ngưỡng {threshold_input:.0%}" )
st.markdown("<br>", unsafe_allow_html=True)
# ==============================================================================
# FOOTER & MODEL EVALUATION SUMMARY
# ==============================================================================
st.divider()
f1, f2, f3, f4 = st.columns(4)
f1.caption(f"📁 Dữ liệu: CDC BRFSS ({len(data):,} hồ sơ)")
f2.caption(f"🧠 Thuật toán: {meta['model_name']}")
f3.caption(
    f"🎯 Chỉ số Test: ROC-AUC {meta['evaluation']['roc_auc']:.3f} | "
    f"PR-AUC {meta['evaluation']['pr_auc']:.3f}"
)
f4.caption(f"⚖️ Ngưỡng tối ưu: {THR * 100:.0f}%")
