import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONFIG & PATHS
# ==============================================================================
BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "final" / "heart_disease_health_indicators_BRFSS2015_V2.csv"
MODEL = BASE / "src" / "model" / "XGBoost" / "model_detail" / "xgboost_model.pkl"
META = BASE / "src" / "model" / "XGBoost" / "model_detail" / "model_metadata.json"
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
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #374151;
        color: #e5e7eb;
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

# ==============================================================================
# DATA LOADING & MODEL CACHING
# ==============================================================================
@st.cache_data
def load():
    d = pd.read_csv(CSV)
    # Cân chỉnh mã giới tính (BRFSS: 1=Nam, 2=Nữ; chuẩn mô hình: 1=Nam, 0=Nữ)
    if 2.0 in d.Sex.values or 2 in d.Sex.values:
        d["Sex"] = d["Sex"].map({2.0: 0, 1.0: 1, 2: 0, 1: 1}).fillna(0).astype(int)
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
    "ver": 0
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def reset():
    ver = st.session_state.ver + 1
    st.session_state.update(DEFAULTS)
    st.session_state.ver = ver

def clear_selection():
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

    st.markdown("### 🏃 Cấp 4: Thói quen Lối sống & Thể trạng")
    st.selectbox("Hút thuốc", ["Tất cả", "Không", "Có"], key="f_smk")
    st.selectbox("Vận động thể chất", ["Tất cả", "Không", "Có"], key="f_act")
    st.slider("Chỉ số khối cơ thể (BMI)", 12.0, 60.0, step=0.5, key="f_bmi")

    st.divider()
    st.button("↻ Đặt lại tất cả bộ lọc", use_container_width=True, on_click=reset)
    
    st.caption("TIẾN TRÌNH ĐỘ PHỦ DỮ LIỆU")

def yn(d, col, v):
    return d if v == "Tất cả" else d[d[col] == (1 if v == "Có" else 0)]

S = st.session_state
base = data

# Apply multi-level filters
if S.f_state != "Tất cả các bang":
    base = base[base.StateName == S.f_state]
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

base = yn(yn(yn(yn(base, "HighBP", S.f_bp), "HighChol", S.f_chol), "Smoker", S.f_smk), "PhysActivity", S.f_act)
base = base[base.BMI.isna() | base.BMI.between(*S.f_bmi)]

with st.sidebar:
    cov_ratio = len(base) / len(data)
    st.progress(cov_ratio)
    st.caption(f"**{len(base):,} / {len(data):,}** bản ghi phù hợp ({cov_ratio * 100:.1f}% tập dữ liệu).")

# ==============================================================================
# CROSS-FILTERING (INTER-CHART LINKAGE)
# ==============================================================================
def picked(key, order):
    ev = st.session_state.get(key)
    if not ev or "selection" not in ev or not ev["selection"].get("points"):
        return []
    return [order[p["point_number"]] for p in ev["selection"]["points"] if p.get("point_number") is not None and p["point_number"] < len(order)]

AGE_KEY, SEX_KEY = f"age_{S.ver}", f"sex_{S.ver}"
age_sel = picked(AGE_KEY, AGE_ORDER)
sex_sel = picked(SEX_KEY, SEX_ORDER)

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
st.markdown('<div class="eyebrow">HỆ THỐNG TRỰC QUAN HÓA THÔNG TIN Y TẾ & SỨC KHỎE DÂN SỐ</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Bảng Điều Khiển Phân Tích & Dự Đoán Nguy Cơ Tim Mạch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Khám phá toàn diện phân bố không gian địa lý, tương quan nhân khẩu học, '
    'bệnh lý lâm sàng, lối sống và phân tầng nguy cơ tim mạch dựa trên Machine Learning. '
    'Hỗ trợ lọc đa cấp, Drill-down chuyên sâu và Cross-filtering (nhấp vào cột/lát cắt để lọc chéo).</div>',
    unsafe_allow_html=True
)

if len(df) == 0:
    st.warning("⚠️ Không có bản ghi nào phù hợp với điều kiện bộ lọc hiện tại. Vui lòng nới lỏng tiêu chí lọc.")
    st.stop()

if age_sel or sex_sel:
    c1, c2 = st.columns([5, 1])
    active_filters = []
    if age_sel:
        active_filters.append(f"Nhóm tuổi: {', '.join(age_sel)}")
    if sex_sel:
        active_filters.append(f"Giới tính: {', '.join(sex_sel)}")
    c1.info("🔍 **Đang lọc chéo (Cross-filter):** " + " | ".join(active_filters))
    c2.button("✕ Bỏ lọc chéo", on_click=clear_selection, use_container_width=True)

# Executive KPI Metrics
k = st.columns(5)
curr_rate = df[TARGET].mean() * 100
delta_nat = curr_rate - NATIONAL_RATE

k[0].metric("Tổng số bản ghi", f"{len(df):,}", help="Số người tham gia khảo sát phù hợp bộ lọc")
k[1].metric(
    "Tỷ lệ bệnh tim mạch",
    f"{curr_rate:.1f}%",
    delta=f"{delta_nat:+.1f}% so với toàn quốc",
    delta_color="inverse",
    help="Tỷ lệ người mắc bệnh tim hoặc từng đau tim trong mẫu"
)
k[2].metric("Nhóm tuổi chiếm ưu thế", str(df.Age_lbl.value_counts().idxmax()), help="Nhóm độ tuổi có số lượng người đông nhất")
k[3].metric("Chỉ số BMI trung bình", f"{df.BMI.mean():.1f}", help="Chỉ số khối cơ thể trung bình của nhóm")
k[4].metric("Tỷ lệ tăng huyết áp", f"{df.HighBP.mean() * 100:.1f}%", help="Tỷ lệ người có tiền sử huyết áp cao")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: GEOGRAPHIC SPATIAL DISTRIBUTION (MAP REQUIREMENT)
# ==============================================================================
with st.container(border=True):
    st.markdown("### 1. 🗺️ Bản đồ Địa lý: Phân bố Không gian Nguy cơ Tim mạch")
    st.caption("Biểu đồ Choropleth Map thể hiện sự chênh lệch tỷ lệ mắc bệnh tim mạch trên các bang của Hoa Kỳ. Hỗ trợ xem phân bố toàn cảnh và Drill-down theo từng bang.")

    state_agg = df.groupby("StateName", observed=True)[TARGET].agg(
        TongSo="count",
        SoCa="sum",
        TyLe="mean"
    ).reset_index()
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
        st.plotly_chart(fig_map, use_container_width=True)

    with drill_col:
        st.markdown("##### 📍 Phân tích & Xếp hạng Bang")
        
        if S.f_state != "Tất cả các bang":
            # State Drill-down details
            st.success(f"**Drill-down:** Đang xem bang **{S.f_state}**")
            st_data = state_agg[state_agg.StateName == S.f_state]
            if len(st_data) > 0:
                st_r = st_data.iloc[0]
                st.metric("Tỷ lệ tại bang", f"{st_r['TyLe_pct']:.1f}%", delta=f"{st_r['TyLe_pct'] - NATIONAL_RATE:+.1f}% vs Toàn quốc", delta_color="inverse")
                st.metric("Số ca bệnh tim", f"{st_r['SoCa']:,} / {st_r['TongSo']:,}")
            else:
                st.info("Không có dữ liệu cho bang này sau khi lọc.")
        else:
            rank_tab1, rank_tab2 = st.tabs(["🔴 Top 5 Cao nhất", "🟢 Top 5 Thấp nhất"])
            with rank_tab1:
                top_high = state_agg.sort_values(by="TyLe_pct", ascending=False).head(5)
                for _, r in top_high.iterrows():
                    st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({r['SoCa']:,} ca)")
            with rank_tab2:
                top_low = state_agg.sort_values(by="TyLe_pct", ascending=True).head(5)
                for _, r in top_low.iterrows():
                    st.markdown(f"**{r['StateName']}**: `{r['TyLe_pct']:.1f}%` ({r['SoCa']:,} ca)")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: DEMOGRAPHICS & CROSS-FILTERING (CHARTS 2 & 3)
# ==============================================================================
st.markdown("### 2. 👥 Nhân khẩu học & Liên kết Lọc chéo (Cross-filtering)")
st.caption("Nhấp chuột vào một cột Nhóm tuổi hoặc lát cắt Giới tính để kích hoạt liên kết lọc chéo toàn bộ dữ liệu trên bảng điều khiển.")

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
    st.markdown("**Biểu đồ Tròn: Phân bổ ca bệnh tim theo Giới tính** `[Cross-filter]`")
    st.caption("Bấm chọn lát cắt để lọc chéo theo giới tính")
    d_sex = by_age(base)
    cases_sex = d_sex[d_sex[TARGET] == 1].Sex_lbl.value_counts().reindex(SEX_ORDER).fillna(0)
    fig_sex = go.Figure(
        go.Pie(
            labels=SEX_ORDER,
            values=cases_sex.values,
            hole=0.58,
            sort=False,
            marker=dict(colors=["#f472b6", "#38bdf8"]),
            textinfo="label+percent",
            hovertemplate="<b>Giới tính: %{label}</b><br>Số ca bệnh tim: %{value:,}<br>Tỷ trọng: %{percent}<extra></extra>"
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
st.markdown("### 3. 🩺 Yếu tố Lâm sàng & Xu hướng Sinh trắc học")
r2a, r2b = st.columns(2)

with r2a.container(border=True):
    # Chart 4: Line Chart with Markers
    st.markdown("**Biểu đồ Đường: Xu hướng nguy cơ theo Độ tuổi × Tiền sử Cao huyết áp**")
    st.caption("So sánh tốc độ gia tăng nguy cơ tim mạch theo tuổi thọ giữa người huyết áp cao và bình thường.")
    g_line = (
        df.dropna(subset=["HighBP"])
        .assign(BP=lambda x: x.HighBP.map({0: "Huyết áp bình thường", 1: "Huyết áp cao"}))
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
    fig_line.update_layout(legend_title_text="Tình trạng Huyết áp")
    st.plotly_chart(sty(fig_line, 330), use_container_width=True)

with r2b.container(border=True):
    # Charts 5 & 6: Histogram & Box Plot with Tabs Drill-down
    st.markdown("**Phân tích Thể trạng: Chỉ số BMI theo Tình trạng Bệnh tim**")
    tab_dist, tab_box = st.tabs(["📊 Phân bố tần suất (Histogram)", "📦 Biểu đồ Hộp & Ngoại lai (Box Plot)"])
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
            xaxis_title="Chỉ số khối cơ thể (BMI)"
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
            yaxis_title="BMI (kg/m²)"
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
    st.plotly_chart(yn_bar(df, life_factors, "Tỷ lệ bệnh tim giữa nhóm Có vs Không theo lối sống"), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: SOCIOECONOMIC TREEMAP & RISK INTERACTION HEATMAP (CHARTS 8, 9, 10)
# ==============================================================================
st.markdown("### 5. 💰 Kinh tế - Xã hội & Ma trận Tương tác Nguy cơ")
r4a, r4b = st.columns(2)

with r4a.container(border=True):
    # Chart 8: Treemap with Hierarchical Drill-down
    st.markdown("**Biểu đồ Cây Phân cấp: Thu nhập × Học vấn (Treemap)** `[Drill-down]`")
    st.caption("Nhấp vào từng khối Thu nhập để đào sâu (Drill-down) vào các bậc Học vấn bên trong. Kích thước = Số bản ghi, Màu = Tỷ lệ bệnh tim.")
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
# SECTION 6: ML PREDICTION & GAUGE INDICATOR (CHARTS 10 & 11)
# ==============================================================================
st.markdown("### 6. 🤖 Phân tầng Nguy cơ Dân số & Dự đoán Cá nhân hóa (XGBoost)")
left_ml, right_ml = st.columns([1, 1])

with left_ml.container(border=True):
    # Chart 10: Risk Histogram Distribution
    st.markdown("**Phân phối Nguy cơ Dự đoán trên Dân số đã chọn**")
    st.caption(f"Đánh giá rủi ro tổng thể do mô hình XGBoost dự đoán (Đường đứt nét màu vàng là ngưỡng sàng lọc {THR:.0%}).")
    p_subset = proba.loc[df.index]
    fig_risk = px.histogram(
        p_subset * 100,
        nbins=50,
        labels={"value": "Xác suất nguy cơ dự đoán (%)"},
        color_discrete_sequence=["#ef4444"]
    )
    fig_risk.add_vline(
        x=THR * 100,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2.5,
        annotation_text="Ngưỡng sàng lọc (20%)",
        annotation_position="top right"
    )
    fig_risk.update_layout(showlegend=False, yaxis_title="Số lượng người")
    st.plotly_chart(sty(fig_risk, 280), use_container_width=True)
    
    high_risk_ratio = (p_subset >= THR).mean() * 100
    st.metric(
        "Tỷ lệ thuộc Nhóm Nguy cơ Cao (Cần can thiệp)",
        f"{high_risk_ratio:.1f}%",
        help="Tỷ lệ người có xác suất dự đoán vượt ngưỡng 20%"
    )

with right_ml.container(border=True):
    st.markdown("**Công cụ Tính điểm Nguy cơ Cá nhân (Interactive Prediction Form)**")
    with st.form("predict_form"):
        fa, fb, fc = st.columns(3)
        p_sex = fa.selectbox("Giới tính", SEX_ORDER)
        p_age = fb.selectbox("Độ tuổi", AGE_ORDER, index=6)
        p_bmi = fc.number_input("Chỉ số BMI", 12.0, 60.0, 27.5, step=0.5)

        p_edu = fa.selectbox("Học vấn", list(EDU_LBL.values()), index=5)
        p_inc = fb.selectbox("Thu nhập", list(INC_LBL.values()), index=6)
        p_dia = fc.selectbox("Tiểu đường", list(DIA_LBL.values()))

        p_gen = fa.select_slider("Sức khỏe tổng quát", list(GEN_LBL.values()), "Tốt")
        p_ment = fb.slider("Số ngày sức khỏe tinh thần kém (30 ngày qua)", 0, 30, 2)
        p_phys = fc.slider("Số ngày thể chất kém (30 ngày qua)", 0, 30, 1)

        t_cols = st.columns(4)
        pred_flags = {
            "HighBP": t_cols[0].toggle("Huyết áp cao", False),
            "HighChol": t_cols[1].toggle("Cholesterol cao", False),
            "CholCheck": t_cols[2].toggle("Đo cholesterol", True),
            "Smoker": t_cols[3].toggle("Hút thuốc", False),
            "Stroke": t_cols[0].toggle("Đã từng đột quỵ", False),
            "DiffWalk": t_cols[1].toggle("Khó khăn khi đi lại", False),
            "PhysActivity": t_cols[2].toggle("Vận động thể dục", True),
            "Fruits": t_cols[3].toggle("Ăn trái cây", True),
            "Veggies": t_cols[0].toggle("Ăn rau xanh", True),
            "HvyAlcoholConsump": t_cols[1].toggle("Uống nhiều rượu", False),
            "AnyHealthcare": t_cols[2].toggle("Có BHYT", True),
            "NoDocbcCost": t_cols[3].toggle("Bỏ khám do chi phí", False),
        }
        submit_btn = st.form_submit_button("⚡ Tính toán Điểm Nguy cơ", type="primary", use_container_width=True)

    if submit_btn:
        row = {
            **{k_: int(v) for k_, v in pred_flags.items()},
            "BMI": p_bmi,
            "Diabetes": list(DIA_LBL.values()).index(p_dia),
            "GenHlth": list(GEN_LBL.values()).index(p_gen) + 1,
            "MentHlth": p_ment,
            "PhysHlth": p_phys,
            "Sex": SEX_ORDER.index(p_sex),
            "Age": AGE_ORDER.index(p_age) + 1,
            "Education": list(EDU_LBL.values()).index(p_edu) + 1,
            "Income": list(INC_LBL.values()).index(p_inc) + 1
        }
        pr = float(model.predict_proba(pd.DataFrame([row])[meta["features"]])[:, 1][0])
        
        # Chart 11: Gauge / Indicator Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pr * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if pr >= THR else "#22c55e", "thickness": 0.8},
                    "steps": [
                        {"range": [0, THR * 100], "color": "rgba(34, 197, 94, 0.2)"},
                        {"range": [THR * 100, 100], "color": "rgba(239, 68, 68, 0.25)"}
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 4},
                        "thickness": 0.85,
                        "value": THR * 100
                    }
                }
            )
        )
        st.plotly_chart(sty(fig_gauge, 190), use_container_width=True)
        if pr >= THR:
            st.error(f"⚠️ **Nhóm Nguy cơ Cao ({pr * 100:.1f}%)**: Xác suất vượt ngưỡng {THR * 100:.0%}. Khuyến nghị kiểm tra chuyên sâu tim mạch.")
        else:
            st.success(f"✅ **Dưới Ngưỡng Nguy cơ ({pr * 100:.1f}%)**: Chỉ số nằm trong vùng kiểm soát an toàn.")
        st.caption("ℹ️ *Kết quả dự đoán có mục đích sàng lọc hỗ trợ quyết định y tế công cộng, không thay thế chẩn đoán bác sĩ.*")

# ==============================================================================
# FOOTER & MODEL EVALUATION SUMMARY
# ==============================================================================
st.divider()
f1, f2, f3, f4 = st.columns(4)
f1.caption(f"📁 Dữ liệu: CDC BRFSS ({len(data):,} hồ sơ)")
f2.caption(f"🧠 Thuật toán: {meta['model_name']}")
f3.caption(f"🎯 ROC-AUC: {meta['test_metrics']['roc_auc']:.3f} | PR-AUC: {meta['test_metrics']['pr_auc']:.3f}")
f4.caption(f"⚖️ Ngưỡng tối ưu: {THR * 100:.0f}%")
