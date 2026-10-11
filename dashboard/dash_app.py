"""CardioView population analytics and cardiovascular risk demo, built with Dash."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/final/heart_disease_health_indicators_BRFSS2015_V2.csv"
MODEL_DIR = ROOT / "src/model/Logisic Regression/model_detail"
MODEL_PATH = MODEL_DIR / "logistic_regression_final.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"
TARGET = "HeartDiseaseorAttack"

bundle = joblib.load(MODEL_PATH)
model, scaler = bundle["model"], bundle["scaler"]
features, threshold = bundle["features"], float(bundle["threshold"])
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
data = pd.read_csv(CSV)
data[TARGET] = data[TARGET].fillna(0).astype(int)
if data["Sex"].max() == 2:  # BRFSS source encoding: 1=male, 2=female.
    data["Sex"] = (data["Sex"] == 1).astype(int)
data["complete"] = data[features + [TARGET]].notna().all(axis=1)
complete_idx = np.flatnonzero(data["complete"].to_numpy())
_, test_idx = train_test_split(
    complete_idx,
    test_size=0.2,
    random_state=42,
    stratify=data.loc[complete_idx, TARGET].to_numpy(),
)
data["is_test"] = np.isin(np.arange(len(data)), test_idx)
data["risk_score"] = np.nan
complete = data.loc[data["complete"]]
data.loc[complete.index, "risk_score"] = model.predict_proba(
    scaler.transform(complete[features])
)[:, 1]

AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+",
}
AGE_ORDER = list(AGE_LABELS.values())
SEX_LABELS = {0: "Nữ", 1: "Nam"}
GEN_LABELS = {1: "Xuất sắc", 2: "Rất tốt", 3: "Tốt", 4: "Trung bình", 5: "Kém"}
DIABETES_LABELS = {0: "Không", 1: "Tiền tiểu đường", 2: "Có"}
INCOME_LABELS = {
    1: "Dưới $10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k",
    5: "$25–35k", 6: "$35–50k", 7: "$50–75k", 8: "$75k+",
}
EDUCATION_LABELS = {
    1: "Chưa đi học", 2: "Tiểu học", 3: "THPT dở dang",
    4: "Tốt nghiệp THPT", 5: "Đại học dở dang", 6: "Tốt nghiệp đại học",
}
STATE_CODES = {
    "Alabama":"AL", "Alaska":"AK", "Arizona":"AZ", "Arkansas":"AR", "California":"CA",
    "Colorado":"CO", "Connecticut":"CT", "Delaware":"DE", "District of Columbia":"DC",
    "Florida":"FL", "Georgia":"GA", "Hawaii":"HI", "Idaho":"ID", "Illinois":"IL",
    "Indiana":"IN", "Iowa":"IA", "Kansas":"KS", "Kentucky":"KY", "Louisiana":"LA",
    "Maine":"ME", "Maryland":"MD", "Massachusetts":"MA", "Michigan":"MI", "Minnesota":"MN",
    "Mississippi":"MS", "Missouri":"MO", "Montana":"MT", "Nebraska":"NE", "Nevada":"NV",
    "New Hampshire":"NH", "New Jersey":"NJ", "New Mexico":"NM", "New York":"NY",
    "North Carolina":"NC", "North Dakota":"ND", "Ohio":"OH", "Oklahoma":"OK", "Oregon":"OR",
    "Pennsylvania":"PA", "Rhode Island":"RI", "South Carolina":"SC", "South Dakota":"SD",
    "Tennessee":"TN", "Texas":"TX", "Utah":"UT", "Vermont":"VT", "Virginia":"VA",
    "Washington":"WA", "West Virginia":"WV", "Wisconsin":"WI", "Wyoming":"WY",
    "Puerto Rico":"PR", "Guam":"GU",
}

COLORS = {"navy": "#122238", "blue": "#1687f8", "teal": "#1ab49b", "red": "#ef5d68",
          "ink": "#172b4d", "muted": "#718198", "line": "#e7edf4", "bg": "#f5f8fc"}
NEG = "Không có tiền sử"
POS = "Có tiền sử bệnh tim/đau tim"


def card(children, class_name="card"):
    return html.Div(children, className=class_name)


def section_heading(title, subtitle=None):
    parts = [html.H3(title, className="section-title")]
    if subtitle:
        parts.append(html.P(subtitle, className="section-subtitle"))
    return html.Div(parts, className="section-heading")


def metric(label, value, detail, accent="blue", identifier=None):
    return html.Div([
        html.Div(className=f"metric-accent {accent}"),
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value", id=identifier),
        html.Div(detail, className="metric-detail"),
    ], className="metric-card")


def dropdown(label, identifier, options, value, clearable=False):
    return html.Div([
        html.Label(label, htmlFor=identifier, className="filter-label"),
        dcc.Dropdown(id=identifier, options=options, value=value, clearable=clearable,
                     searchable=True, className="filter-dropdown"),
    ], className="filter-group")


def app_shell():
    states = sorted(data["StateName"].dropna().unique())
    return html.Div([
        html.Aside([
            html.Div([html.Div("C", className="brand-mark"), html.Div([
                html.Div("CardioView", className="brand-name"),
                html.Div("HEALTH INTELLIGENCE", className="brand-kicker"),
            ])], className="brand-row"),
            html.Div("KHÁM PHÁ DỮ LIỆU", className="sidebar-label"),
            dropdown("Tập dữ liệu", "filter-set", [
                {"label":"Toàn bộ dữ liệu", "value":"all"},
                {"label":"Tập test độc lập", "value":"test"}], "all"),
            dropdown("Bang / vùng lãnh thổ", "filter-state",
                     [{"label":"Tất cả", "value":"all"}] + [{"label":s, "value":s} for s in states], "all"),
            dropdown("Giới tính", "filter-sex", [{"label":"Tất cả", "value":"all"},
                     {"label":"Nữ", "value":0}, {"label":"Nam", "value":1}], "all"),
            dropdown("Nhóm tuổi", "filter-age", [{"label":"Tất cả nhóm tuổi", "value":"all"}] +
                     [{"label":v, "value":k} for k, v in AGE_LABELS.items()], "all"),
            dropdown("Huyết áp cao", "filter-bp", [{"label":"Tất cả", "value":"all"},
                     {"label":"Không", "value":0}, {"label":"Có", "value":1}], "all"),
            dropdown("Tiểu đường", "filter-diabetes", [{"label":"Tất cả", "value":"all"}] +
                     [{"label":v, "value":k} for k, v in DIABETES_LABELS.items()], "all"),
            html.Div([html.Label("Khoảng BMI", className="filter-label"),
                      dcc.RangeSlider(12, 60, 0.5, value=[12, 60], id="filter-bmi",
                                      marks={12:"12", 25:"25", 40:"40", 60:"60"},
                                      tooltip={"placement":"bottom", "always_visible":False})],
                     className="filter-group bmi-filter"),
            html.Button("Đặt lại bộ lọc", id="reset-filters", className="reset-button", n_clicks=0),
            html.Div(className="sidebar-spacer"),
            html.Div([html.Span("●", className="live-dot"), " Dữ liệu BRFSS 2015"], className="source-note"),
        ], className="sidebar"),
        html.Main([
            html.Header([
                html.Div([html.Div("CARDIOVASCULAR HEALTH · BRFSS 2015", className="eyebrow"),
                          html.H1("Tổng quan sức khỏe tim mạch", className="page-title"),
                          html.P("Khám phá dữ liệu dân số, đánh giá mô hình và thử dự đoán nguy cơ cá nhân.",
                                 className="page-subtitle")]),
                html.Div([html.Div("●", className="status-dot"), html.Span("Mô hình sẵn sàng")],
                         className="model-status")
            ], className="topbar"),
            dcc.Tabs(id="main-tabs", value="overview", className="main-tabs", children=[
                dcc.Tab(label="Tổng quan", value="overview", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Dự đoán cá nhân", value="prediction", className="tab", selected_className="tab-selected"),
            ]),
            html.Div(id="tab-content", className="tab-content"),
            html.Footer([html.Span("CardioView · Phân tích dữ liệu khảo sát dân số"),
                         html.Span("Kết quả dự đoán chỉ dùng cho mục đích tham khảo, không thay thế chẩn đoán y khoa.")],
                        className="footer"),
        ], className="main-content"),
    ], className="app-shell")


app = Dash(__name__, title="CardioView · Sức khỏe tim mạch", update_title="Đang cập nhật…",
           suppress_callback_exceptions=True)
app.layout = app_shell
server = app.server


def overview_layout():
    return html.Div([
        html.Div([
            metric("SỐ HỒ SƠ PHÙ HỢP", "—", "Sau khi áp dụng bộ lọc", "blue", "metric-total"),
            metric("TỶ LỆ CÓ TIỀN SỬ", "—", "Bệnh tim hoặc đau tim", "red", "metric-rate"),
            metric("BMI TRUNG BÌNH", "—", "kg/m² trong nhóm đang xem", "teal", "metric-bmi"),
            metric("NHÓM TUỔI PHỔ BIẾN", "—", "Nhóm có nhiều hồ sơ nhất", "violet", "metric-age"),
        ], className="metric-grid"),
        html.Div(id="filter-summary", className="filter-summary"),
        html.Div([
            card([html.Div([html.H3("Tỷ lệ theo tuổi", className="chart-title"),
                            html.P("Tỷ lệ có tiền sử bệnh tim/đau tim theo nhóm tuổi", className="chart-hint")]),
                  dcc.Graph(id="age-chart", config={"displayModeBar":False})]),
            card([html.Div([html.H3("Phân bố theo bang", className="chart-title"),
                            html.P("Các bang có tỷ lệ cao nhất trong nhóm đang chọn", className="chart-hint")]),
                  dcc.Graph(id="state-chart", config={"displayModeBar":False})]),
        ], className="chart-grid two-col"),
        html.Div([
            card([html.Div([html.H3("Yếu tố sức khỏe", className="chart-title"),
                            html.P("Tỷ lệ có tiền sử theo tình trạng sức khỏe", className="chart-hint")]),
                  dcc.Graph(id="factor-chart", config={"displayModeBar":False})]),
            card([html.Div([html.H3("BMI và tiền sử bệnh", className="chart-title"),
                            html.P("Phân bố BMI của hai nhóm khảo sát", className="chart-hint")]),
                  dcc.Graph(id="bmi-chart", config={"displayModeBar":False})]),
        ], className="chart-grid two-col"),
    ])


def prediction_layout():
    def select(label, ident, opts, val):
        return dropdown(label, ident, [{"label":a, "value":b} for a, b in opts], val)
    binary = [("Không", 0), ("Có", 1)]
    return html.Div([
        section_heading("Ước tính nguy cơ cá nhân", "Nhập thông tin khảo sát để xem điểm xác suất từ mô hình Logistic Regression."),
        html.Div([
            card([html.Div("01", className="form-step"), html.H3("Sức khỏe & tiền sử", className="form-title"),
                  html.Div([
                      select("Huyết áp cao", "p-bp", binary, 0), select("Cholesterol cao", "p-chol", binary, 0),
                      html.Div([html.Label("BMI", className="filter-label"), dcc.Input(id="p-bmi", type="number", min=12, max=60, step=0.1, value=25, className="number-input")], className="filter-group"),
                      select("Tiểu đường", "p-diabetes", [(v,k) for k,v in DIABETES_LABELS.items()], 0),
                      select("Sức khỏe tổng quát", "p-gen", [(v,k) for k,v in GEN_LABELS.items()], 3),
                      select("Nhóm tuổi", "p-age", [(v,k) for k,v in AGE_LABELS.items()], 7),
                      html.Div([html.Label("Ngày sức khỏe tinh thần không tốt", className="filter-label"), dcc.Slider(0,30,1,value=0,id="p-mental",marks={0:"0",15:"15",30:"30"})], className="filter-group full-width"),
                      html.Div([html.Label("Ngày sức khỏe thể chất không tốt", className="filter-label"), dcc.Slider(0,30,1,value=0,id="p-physical",marks={0:"0",15:"15",30:"30"})], className="filter-group full-width"),
                      select("Tiền sử đột quỵ", "p-stroke", binary, 0), select("Khó khăn khi đi lại", "p-walk", binary, 0),
                  ], className="form-grid")]),
            card([html.Div("02", className="form-step teal-step"), html.H3("Lối sống & nhân khẩu học", className="form-title"),
                  html.Div([
                      select("Hút thuốc (≥100 điếu trong đời)", "p-smoker", binary, 0), select("Hoạt động thể chất", "p-activity", binary, 1),
                      select("Ăn trái cây ít nhất mỗi ngày", "p-fruit", binary, 1), select("Ăn rau ít nhất mỗi ngày", "p-veggie", binary, 1),
                      select("Uống rượu bia nhiều", "p-alcohol", binary, 0), select("Giới tính", "p-sex", [("Nữ",0),("Nam",1)], 0),
                      select("Trình độ học vấn", "p-education", [(v,k) for k,v in EDUCATION_LABELS.items()], 6),
                      select("Thu nhập hộ gia đình", "p-income", [(v,k) for k,v in INCOME_LABELS.items()], 7),
                  ], className="form-grid"),
                  html.Div([html.Label("Ngưỡng phân loại", className="filter-label"),
                            dcc.Slider(10,90,1,value=round(threshold*100),id="p-threshold",marks={10:"10%",threshold*100:f"Mặc định {threshold:.0%}",90:"90%"},
                                       tooltip={"placement":"bottom","always_visible":False})], className="filter-group threshold-control"),
                  html.Button(["Tính điểm nguy cơ", html.Span(" →", className="button-arrow")], id="predict-button", n_clicks=0, className="primary-button"),
            ]),
        ], className="chart-grid two-col prediction-grid"),
        html.Div(id="prediction-result", className="prediction-result"),
    ])


def plot_layout(height=340):
    return dict(template="plotly_white", height=height, margin=dict(l=18,r=18,t=18,b=30),
                paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Inter, sans-serif", color=COLORS["ink"], size=12),
                xaxis=dict(showgrid=False, zeroline=False, linecolor=COLORS["line"]),
                yaxis=dict(gridcolor="#eef2f7", zeroline=False, linecolor=COLORS["line"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "prediction":
        return prediction_layout()
    return overview_layout()


@app.callback(
    Output("filter-set", "value"), Output("filter-state", "value"), Output("filter-sex", "value"),
    Output("filter-age", "value"), Output("filter-bp", "value"), Output("filter-diabetes", "value"),
    Output("filter-bmi", "value"), Input("reset-filters", "n_clicks"), prevent_initial_call=True,
)
def reset_filters(_):
    return "all", "all", "all", "all", "all", "all", [12, 60]


@app.callback(
    Output("metric-total", "children"), Output("metric-rate", "children"), Output("metric-bmi", "children"),
    Output("metric-age", "children"), Output("filter-summary", "children"),
    Output("age-chart", "figure"), Output("state-chart", "figure"), Output("factor-chart", "figure"),
    Output("bmi-chart", "figure"),
    Input("filter-set", "value"), Input("filter-state", "value"), Input("filter-sex", "value"),
    Input("filter-age", "value"), Input("filter-bp", "value"), Input("filter-diabetes", "value"),
    Input("filter-bmi", "value"),
)
def update_overview(dataset_set, state, sex, age, bp, diabetes, bmi_range):
    d = data[data["is_test"]] if dataset_set == "test" else data
    if sex != "all": d = d[d["Sex"] == sex]
    if age != "all": d = d[d["Age"] == age]
    if bp != "all": d = d[d["HighBP"] == bp]
    if diabetes != "all": d = d[d["Diabetes"] == diabetes]
    d = d[d["BMI"].isna() | d["BMI"].between(bmi_range[0], bmi_range[1])]
    geo = d.copy()
    if state != "all": d = d[d["StateName"] == state]
    if len(d) == 0:
        empty = go.Figure().update_layout(**plot_layout(), annotations=[dict(text="Không có hồ sơ phù hợp", showarrow=False, x=.5, y=.5, xref="paper", yref="paper")])
        return "0", "—", "—", "—", "Không có hồ sơ cho bộ lọc hiện tại.", empty, empty, empty, empty

    rate = d[TARGET].mean() * 100
    age_counts = d.groupby("Age", observed=False)[TARGET].agg(["mean", "count"]).reset_index()
    age_counts["age_label"] = age_counts["Age"].map(AGE_LABELS)
    fig_age = px.bar(age_counts, x="age_label", y="mean", custom_data=["count"],
                     color="mean", color_continuous_scale=[[0,"#b8e5df"],[1,"#1687f8"]],
                     labels={"age_label":"Nhóm tuổi", "mean":"Tỷ lệ"})
    fig_age.update_traces(text=age_counts["mean"].map(lambda x:f"{x:.1%}"), textposition="outside",
                          hovertemplate="%{x}<br>Tỷ lệ: %{y:.1%}<br>Số hồ sơ: %{customdata[0]:,}<extra></extra>")
    fig_age.update_layout(**plot_layout(), coloraxis_showscale=False, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Tỷ lệ")

    state_rates = geo.groupby("StateName", observed=True)[TARGET].agg(["mean", "count"]).reset_index().sort_values("mean").tail(12)
    fig_state = px.bar(state_rates, x="mean", y="StateName", orientation="h", custom_data=["count"],
                       color="mean", color_continuous_scale=[[0,"#f8c9cc"],[1,"#ef5d68"]],
                       labels={"mean":"Tỷ lệ", "StateName":"Bang"})
    fig_state.update_traces(text=state_rates["mean"].map(lambda x:f"{x:.1%}"), textposition="outside",
                            hovertemplate="%{y}<br>Tỷ lệ: %{x:.1%}<br>Số hồ sơ: %{customdata[0]:,}<extra></extra>")
    fig_state.update_layout(**plot_layout(), coloraxis_showscale=False, xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Tỷ lệ")

    factors = {"Huyết áp cao": "HighBP", "Cholesterol cao": "HighChol", "Đột quỵ": "Stroke", "Khó đi lại": "DiffWalk", "Tiểu đường": "Diabetes"}
    rows=[]
    for label, col in factors.items():
        for val, group in d.groupby(col, observed=True):
            if col == "Diabetes": category=DIABETES_LABELS.get(int(val), str(val))
            else: category="Có" if val else "Không"
            rows.append({"Yếu tố":label,"Nhóm":category,"Tỷ lệ":group[TARGET].mean()*100,"n":len(group)})
    factor_df=pd.DataFrame(rows)
    fig_factor=px.bar(factor_df,x="Tỷ lệ",y="Yếu tố",color="Nhóm",barmode="group",orientation="h",custom_data=["n"],
                       color_discrete_sequence=["#1687f8","#ef5d68","#1ab49b"],labels={"Tỷ lệ":"Tỷ lệ (%)"})
    fig_factor.update_traces(hovertemplate="%{y} · %{fullData.name}<br>Tỷ lệ: %{x:.1f}%<br>Số hồ sơ: %{customdata[0]:,}<extra></extra>")
    fig_factor.update_layout(**plot_layout(380),yaxis_title="",xaxis_title="Tỷ lệ (%)",legend_title="Tình trạng")

    bmi_df=d.dropna(subset=["BMI"]).copy(); bmi_df["Nhóm"] = bmi_df[TARGET].map({0:NEG,1:POS})
    fig_bmi=px.histogram(bmi_df,x="BMI",color="Nhóm",nbins=36,barmode="overlay",histnorm="percent",opacity=.72,
                         color_discrete_map={NEG:"#1687f8",POS:"#ef5d68"},labels={"BMI":"BMI (kg/m²)","percent":"Tỷ trọng (%)"})
    fig_bmi.update_layout(**plot_layout(),xaxis_title="BMI (kg/m²)",yaxis_title="Tỷ trọng (%)",legend_title="")
    summary = f"{len(d):,} hồ sơ đang hiển thị · tỷ lệ nền toàn bộ mẫu {data[TARGET].mean():.1%}"
    return f"{len(d):,}", f"{rate:.1f}%", f"{d['BMI'].mean():.1f}", AGE_LABELS[int(d['Age'].mode().iloc[0])], summary, fig_age, fig_state, fig_factor, fig_bmi


PREDICT_INPUTS = [Input("predict-button", "n_clicks"), Input("p-bp", "value"), Input("p-chol", "value"),
                  Input("p-bmi", "value"), Input("p-diabetes", "value"), Input("p-gen", "value"), Input("p-age", "value"),
                  Input("p-mental", "value"), Input("p-physical", "value"), Input("p-stroke", "value"), Input("p-walk", "value"),
                  Input("p-smoker", "value"), Input("p-activity", "value"), Input("p-fruit", "value"), Input("p-veggie", "value"),
                  Input("p-alcohol", "value"), Input("p-sex", "value"), Input("p-education", "value"), Input("p-income", "value"),
                  Input("p-threshold", "value")]


@app.callback(Output("prediction-result", "children"), *PREDICT_INPUTS, prevent_initial_call=True)
def predict_person(n_clicks, high_bp, high_chol, bmi, diabetes, gen_health, age, mental, physical,
                   stroke, walk, smoker, activity, fruit, veggie, alcohol, sex, education, income, threshold_pct):
    if not n_clicks:
        return ""
    row = {"HighBP":high_bp, "HighChol":high_chol, "CholCheck":1, "BMI":bmi, "Smoker":smoker,
           "Stroke":stroke, "Diabetes":diabetes, "PhysActivity":activity, "Fruits":fruit, "Veggies":veggie,
           "HvyAlcoholConsump":alcohol, "AnyHealthcare":1, "NoDocbcCost":0, "GenHlth":gen_health,
           "MentHlth":mental, "PhysHlth":physical, "DiffWalk":walk, "Sex":sex, "Age":age,
           "Education":education, "Income":income}
    x = pd.DataFrame([row])[features]
    probability = float(model.predict_proba(scaler.transform(x))[0,1])
    risk = probability * 100
    chosen_threshold = float(threshold_pct) / 100
    high = probability >= chosen_threshold
    color = COLORS["red"] if high else COLORS["teal"]
    return card([
        html.Div([html.Div("MÔ HÌNH LOGISTIC REGRESSION", className="eyebrow"),
                  html.H2("Kết quả ước tính", className="result-title"),
                  html.P("Xác suất thuộc nhóm có báo cáo bệnh tim hoặc từng đau tim", className="chart-hint")]),
        html.Div([html.Div(f"{risk:.1f}%", className="result-score", style={"color":color}),
                  html.Div([html.Div(style={"width":f"{risk:.1f}%","background":color},className="risk-fill")],className="risk-track"),
                  html.Div([html.Span("0%"),html.Span("50%"),html.Span("100%")],className="risk-scale")],className="risk-score-wrap"),
        html.Div("Ngưỡng phân loại: " + f"{chosen_threshold:.0%}", className="threshold-note"),
        html.Div("Mô hình xếp hồ sơ vào nhóm nguy cơ cao tại ngưỡng đã chọn." if high else "Mô hình xếp hồ sơ dưới ngưỡng phân loại đã chọn.",
                 className="risk-verdict", style={"borderColor":color,"color":color}),
        html.P("Điểm số là kết quả sàng lọc thống kê từ khảo sát dân số, không phải xác suất chẩn đoán cá nhân. Không thay thế tư vấn của chuyên gia y tế.", className="medical-note"),
    ], "result-card")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
