import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from real_estate_utils import (
    ALL_FEATURES,
    BASE_GROWTH_RATE,
    CURRENT_YEAR,
    HORIZON_YEARS,
    engineer_features,
)

st.set_page_config(
    page_title="Real Estate Investment Advisor", page_icon="🏙️", layout="wide"
)


# ---- bigger fonts (change BASE_FONT_PX to taste) ----
BASE_FONT_PX = 22
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.2rem; }}
html, body, .stApp {{ font-size: {BASE_FONT_PX}px; }}

/* text, labels, inputs, buttons, tables */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stCaptionContainer"] p, [data-testid="stAlertContainer"] p,
[data-testid="stTable"] td, [data-testid="stTable"] th,
.stButton button p, .stFormSubmitButton button p, .stDownloadButton button p,
[data-baseweb="select"] div, [data-baseweb="input"] input, [data-baseweb="tab"] p,
[data-baseweb="tag"] span, .stNumberInput input, .stCheckbox label p {{
    font-size: {BASE_FONT_PX}px !important;
}}
[data-testid="stMetricValue"] {{ font-size: {BASE_FONT_PX + 18}px !important; }}
[data-testid="stMetricLabel"] p {{ font-size: {BASE_FONT_PX}px !important; }}
[data-testid="stMetricDelta"] {{ font-size: {BASE_FONT_PX - 3}px !important; }}
h1 {{ font-size: {BASE_FONT_PX + 24}px !important; }}
h2, h3 {{ font-size: {BASE_FONT_PX + 10}px !important; }}

/* ---- hero banner ---- */
.hero {{
    background: linear-gradient(120deg, #0f1f4d 0%, #1f3a8a 55%, #3b6fe0 100%);
    border-radius: 22px; padding: 30px 42px; margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(15, 31, 77, .35);
}}
.hero-title {{ font-size: {BASE_FONT_PX + 22}px; font-weight: 800; color: #ffffff !important; line-height: 1.2; }}
.hero-sub {{ font-size: {BASE_FONT_PX}px; color: #d6e4ff !important; margin: 10px 0 18px 0; }}
.hero-pill {{
    display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px; padding: 7px 20px; color: #ffffff !important; font-weight: 700; font-size: {BASE_FONT_PX - 4}px;
}}
.hero-pill .dot {{
    display: inline-block; width: 11px; height: 11px; border-radius: 50%; background: #2ee59d; margin-right: 10px;
}}

/* ---- tab-style navigation (underline on the active page) ---- */
.st-key-tabnav div[role="radiogroup"] {{
    gap: 4px; flex-wrap: wrap; border-bottom: 2px solid rgba(128,128,128,.35); margin-bottom: 10px;
}}
.st-key-tabnav div[role="radiogroup"] > label {{
    padding: 8px 18px; margin: 0 0 -2px 0; cursor: pointer; border-bottom: 3px solid transparent;
}}
.st-key-tabnav div[role="radiogroup"] > label > div:first-child {{ display: none; }}   /* hide radio dot */
.st-key-tabnav div[role="radiogroup"] > label:hover {{ background: rgba(128,128,128,.12); }}
.st-key-tabnav div[role="radiogroup"] > label:has(input:checked) {{ border-bottom-color: #ff4b4b; }}
.st-key-tabnav div[role="radiogroup"] > label:has(input:checked) p {{ color: #ff4b4b !important; font-weight: 700; }}
</style>
""",
    unsafe_allow_html=True,
)


def show_in(where, fig):
    """Render a plotly figure with a larger font."""
    fig.update_layout(font_size=18, title_font_size=24, legend_font_size=17)
    where.plotly_chart(fig)


BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_FILE = BASE / "data" / "processed_housing.csv"

# Loaders


@st.cache_resource
def load_models():
    return joblib.load(MODEL_DIR / "clf_model.joblib"), joblib.load(
        MODEL_DIR / "reg_model.joblib"
    )


@st.cache_data
def load_json(name):
    with open(MODEL_DIR / name) as f:
        return json.load(f)


@st.cache_data
def load_importance():
    return pd.read_csv(MODEL_DIR / "clf_importance.csv"), pd.read_csv(
        MODEL_DIR / "reg_importance.csv"
    )


@st.cache_data
def load_data():
    cols = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Furnished_Status",
        "Availability_Status",
        "Infrastructure_Score",
        "Good_Investment",
        "Future_Price_5Y_Lakhs",
    ]
    return pd.read_csv(DATA_FILE, usecols=cols)


required = [
    MODEL_DIR / n
    for n in ("clf_model.joblib", "reg_model.joblib", "meta.json", "metrics.json")
] + [DATA_FILE]
if not all(p.exists() for p in required):
    st.error(
        "Model artifacts not found. Run the notebook `real_estate_investment_advisor.ipynb` first "
        "(it creates the `models/` and `data/` folders), then restart this app."
    )
    st.stop()

clf, reg = load_models()
meta = load_json("meta.json")
metrics = load_json("metrics.json")
data = load_data()
fmt_lakh = lambda v: f"₹ {v:,.1f} L" if v < 100 else f"₹ {v / 100:,.2f} Cr"


# Hero banner + tab menu

st.markdown(
    f"""
<div class="hero">
<div class="hero-title">🏙️ Real Estate Investment Advisor</div>
<div class="hero-sub">ML-powered Investment Classification · 5-Year Price Forecasting · Market Analytics</div>
<div class="hero-pill"><span class="dot"></span>{metrics['best_clf']} Intelligence System · Dashboard Online · {len(data):,} properties</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(key="tabnav"):
    page = st.radio(
        "Menu",
        ["🔮 Predict a Property", "📊 Market Explorer", "🧠 Model Insights"],
        horizontal=True,
        label_visibility="collapsed",
    )

# the sidebar is only needed for the Market Explorer filters
if not page.startswith("📊"):
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
        "[data-testid='collapsedControl'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

# Page 1 - Predict


def page_predict():
    st.title("🔮 Is this property a good investment?")
    st.write(
        "Enter the property details to get an investment verdict and a 5-year price forecast."
    )

    cats = meta["categories"]

    # State / City live outside the form so City options update when State changes
    c1, c2 = st.columns(2)
    states = sorted(set(meta["city_state"].values()))
    state = c1.selectbox("State", states)
    cities = sorted(c for c, s in meta["city_state"].items() if s == state)
    city = c2.selectbox("City", cities)

    with st.form("property_form"):
        a, b, c = st.columns(3)
        ptype = a.selectbox("Property type", cats["Property_Type"])
        bhk = b.number_input(
            "BHK", meta["ranges"]["bhk"][0], max(meta["ranges"]["bhk"][1], 6), 3
        )
        size = c.number_input(
            "Size (sq ft)",
            100,
            20000,
            int(np.clip(1500, *meta["ranges"]["size"])),
            step=50,
        )

        a, b, c = st.columns(3)
        price = a.number_input(
            "Current price (₹ Lakhs)",
            1.0,
            100000.0,
            float(round(meta["global_median_price"], 1)),
            step=5.0,
        )
        year_built = b.number_input(
            "Year built", 1950, CURRENT_YEAR + 5, min(2015, CURRENT_YEAR)
        )
        furnished = c.selectbox("Furnished status", cats["Furnished_Status"])

        a, b, c = st.columns(3)
        total_floors = a.number_input("Total floors", 1, 100, 10)
        floor_no = b.number_input("Floor number", 0, 100, 3)
        facing = c.selectbox("Facing", cats["Facing"])

        a, b, c = st.columns(3)
        schools = a.slider(
            "Nearby schools",
            0,
            int(meta["schools_max"]),
            int(meta["ranges"]["schools_median"]),
        )
        hospitals = b.slider(
            "Nearby hospitals",
            0,
            int(meta["hospitals_max"]),
            int(meta["ranges"]["hospitals_median"]),
        )
        transport = c.selectbox(
            "Public transport accessibility", meta["transport_options"]
        )

        a, b, c = st.columns(3)
        owner = a.selectbox("Owner type", cats["Owner_Type"])
        availability = b.selectbox("Availability status", cats["Availability_Status"])
        parking = c.number_input("Parking spaces", 0, 10, 1)

        a, b = st.columns([1, 2])
        security = a.radio(
            "Security (gated / guard / CCTV)", ["Yes", "No"], horizontal=True
        )
        amenities = b.multiselect(
            "Amenities", meta["amenity_options"], default=meta["amenity_options"][:2]
        )

        submitted = st.form_submit_button("Analyse property", type="primary")

    if not submitted:
        return

    row = {
        "State": state,
        "City": city,
        "Locality": "",
        "Property_Type": ptype,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": price * 1e5 / size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": min(floor_no, total_floors),
        "Total_Floors": total_floors,
        "Age_of_Property": max(CURRENT_YEAR - year_built, 0),
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": ", ".join(amenities),
        "Facing": facing,
        "Owner_Type": owner,
        "Availability_Status": availability,
    }
    X = engineer_features(pd.DataFrame([row]), meta)[ALL_FEATURES]

    proba = clf.predict_proba(X)[0]
    p_good = float(proba[1])
    is_good = p_good >= 0.5
    confidence = max(proba)
    future = float(reg.predict(X)[0])
    gain = future / price - 1
    cagr = (future / price) ** (1 / HORIZON_YEARS) - 1
    fixed_future = price * (1 + BASE_GROWTH_RATE) ** HORIZON_YEARS
    rmse = meta.get("test_rmse", 0.0)

    st.divider()
    if is_good:
        st.success(f"✅ **Good Investment** — model confidence {confidence:.0%}")
    else:
        st.warning(
            f"⚠️ **Not a strong investment** — model confidence {confidence:.0%}"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current price", fmt_lakh(price))
    m2.metric(
        f"Estimated price in {HORIZON_YEARS} yrs", fmt_lakh(future), f"{gain:+.1%}"
    )
    m3.metric(
        "Implied annual growth",
        f"{cagr:.2%}",
        f"{(cagr - BASE_GROWTH_RATE) * 100:+.2f} pp vs 8% baseline",
    )
    m4.metric("Price per sq ft", f"₹ {row['Price_per_SqFt']:,.0f}")

    g1, g2 = st.columns(2)
    with g1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_good * 100,
                number={"suffix": "%"},
                title={"text": "Probability of being a Good Investment"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2e86c1"},
                    "steps": [
                        {"range": [0, 50], "color": "#fadbd8"},
                        {"range": [50, 100], "color": "#d5f5e3"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
            )
        )
        gauge.update_layout(height=320, margin=dict(t=60, b=10))
        show_in(st, gauge)
    with g2:
        years = np.arange(0, HORIZON_YEARS + 1)
        fig = go.Figure()
        fig.add_scatter(
            x=years,
            y=price * (1 + cagr) ** years,
            name="Model forecast",
            mode="lines+markers",
        )
        fig.add_scatter(
            x=years,
            y=price * (1 + BASE_GROWTH_RATE) ** years,
            name="Fixed 8% baseline",
            mode="lines",
            line=dict(dash="dash"),
        )
        if rmse:
            fig.add_scatter(
                x=[HORIZON_YEARS, HORIZON_YEARS],
                y=[future - 1.96 * rmse, future + 1.96 * rmse],
                mode="lines",
                line=dict(width=6, color="rgba(231,76,60,.5)"),
                name="≈95% range (test RMSE)",
            )
        fig.update_layout(
            title="Projected value (₹ Lakhs)",
            xaxis_title="Years from now",
            yaxis_title="₹ Lakhs",
            height=320,
            margin=dict(t=60, b=10),
        )
        show_in(st, fig)

    city_med = meta["city_median_ppsf"].get(city, meta["global_median_ppsf"])
    diff = row["Price_per_SqFt"] / city_med - 1
    st.markdown(
        f"**Context:** price/sq ft is **{abs(diff):.0%} {'below' if diff < 0 else 'above'}** the {city} median "
        f"(₹ {city_med:,.0f}). A fixed 8% growth rate would give **{fmt_lakh(fixed_future)}**."
    )
    if rmse:
        st.caption(
            f"Forecast range uses the model's test RMSE (±{1.96 * rmse:,.1f} L). Predictions are estimates, not financial advice."
        )

    clf_imp, reg_imp = load_importance()
    i1, i2 = st.columns(2)
    show_in(
        i1,
        px.bar(
            clf_imp.head(10).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            title="What drives the investment verdict",
        ).update_layout(height=360),
    )
    show_in(
        i2,
        px.bar(
            reg_imp.head(10).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            title="What drives the price forecast",
        ).update_layout(height=360),
    )


# Page 2 - Market explorer


def page_explorer():
    st.title("📊 Market Explorer")

    with st.sidebar:
        st.subheader("Filters")
        f_states = st.multiselect("State", sorted(data["State"].unique()))
        pool = data[data["State"].isin(f_states)] if f_states else data
        f_cities = st.multiselect("City", sorted(pool["City"].unique()))
        f_types = st.multiselect(
            "Property type", sorted(data["Property_Type"].unique())
        )
        f_bhk = st.multiselect("BHK", sorted(data["BHK"].unique()))
        p_lo, p_hi = float(data["Price_in_Lakhs"].min()), float(
            data["Price_in_Lakhs"].max()
        )
        f_price = st.slider("Price (₹ Lakhs)", p_lo, p_hi, (p_lo, p_hi))
        s_lo, s_hi = int(data["Size_in_SqFt"].min()), int(data["Size_in_SqFt"].max())
        f_size = st.slider("Size (sq ft)", s_lo, s_hi, (s_lo, s_hi))
        only_good = st.checkbox("Only 'Good Investment' properties")

    d = data
    if f_states:
        d = d[d["State"].isin(f_states)]
    if f_cities:
        d = d[d["City"].isin(f_cities)]
    if f_types:
        d = d[d["Property_Type"].isin(f_types)]
    if f_bhk:
        d = d[d["BHK"].isin(f_bhk)]
    d = d[d["Price_in_Lakhs"].between(*f_price) & d["Size_in_SqFt"].between(*f_size)]
    if only_good:
        d = d[d["Good_Investment"] == 1]

    if d.empty:
        st.info("No properties match these filters.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Properties", f"{len(d):,}")
    k2.metric("Median price", fmt_lakh(d["Price_in_Lakhs"].median()))
    k3.metric("Median price / sq ft", f"₹ {d['Price_per_SqFt'].median():,.0f}")
    k4.metric("Good investments", f"{d['Good_Investment'].mean():.0%}")

    t1, t2, t3 = st.tabs(["Heatmap", "Trends", "Distributions"])
    with t1:
        top_cities = d["City"].value_counts().head(25).index
        piv = d[d["City"].isin(top_cities)].pivot_table(
            index="City",
            columns="Property_Type",
            values="Price_per_SqFt",
            aggfunc="median",
        )
        show_in(
            st,
            px.imshow(
                piv,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Median price per sq ft - city × property type (top 25 cities)",
                labels=dict(color="₹ / sq ft"),
            ).update_layout(height=650),
        )
        g = d.groupby("City").agg(
            good_rate=("Good_Investment", "mean"), n=("Good_Investment", "size")
        )
        g = (
            g[g["n"] >= 20]
            .sort_values("good_rate", ascending=False)
            .head(15)
            .reset_index()
        )
        show_in(
            st,
            px.bar(
                g,
                x="City",
                y="good_rate",
                title="Share of Good Investments by city (top 15)",
            ).update_layout(yaxis_tickformat=".0%"),
        )
    with t2:
        t = d.assign(Year_Bin=(d["Year_Built"] // 5) * 5)
        top5 = t["City"].value_counts().head(5).index
        tt = (
            t[t["City"].isin(top5)]
            .groupby(["Year_Bin", "City"])["Price_per_SqFt"]
            .median()
            .reset_index()
        )
        show_in(
            st,
            px.line(
                tt,
                x="Year_Bin",
                y="Price_per_SqFt",
                color="City",
                markers=True,
                title="Median price/sq ft by year built (5 biggest cities in selection)",
            ),
        )
        lo = (
            d.groupby("City")["Price_in_Lakhs"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        show_in(
            st,
            px.bar(
                lo, x="City", y="Price_in_Lakhs", title="Average price by city (top 15)"
            ),
        )
    with t3:
        smp = d.sample(min(len(d), 8000), random_state=1)
        c1, c2 = st.columns(2)
        show_in(
            c1,
            px.histogram(
                d, x="Price_in_Lakhs", nbins=50, title="Price distribution (₹ Lakhs)"
            ),
        )
        show_in(
            c2,
            px.scatter(
                smp,
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="Property_Type",
                opacity=0.5,
                title="Size vs price",
            ),
        )
        show_in(
            st,
            px.box(
                d,
                x="Property_Type",
                y="Price_per_SqFt",
                title="Price per sq ft by property type",
            ),
        )
        show_in(
            st,
            (
                px.scatter(
                    smp,
                    x="Infrastructure_Score",
                    y="Price_per_SqFt",
                    opacity=0.4,
                    trendline="ols",
                    title="Infrastructure score vs price per sq ft",
                )
                if _has_statsmodels()
                else px.scatter(
                    smp,
                    x="Infrastructure_Score",
                    y="Price_per_SqFt",
                    opacity=0.4,
                    title="Infrastructure score vs price per sq ft",
                )
            ),
        )

    st.subheader("Matching properties")
    st.dataframe(d.sort_values("Price_in_Lakhs").head(500))
    st.download_button(
        "Download filtered data (CSV)",
        d.to_csv(index=False).encode(),
        "filtered_properties.csv",
        "text/csv",
    )


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401

        return True
    except ImportError:
        return False


# Page 3 - Model insights


def page_insights():
    st.title("🧠 Model Insights")
    st.write(
        "Test-set performance of every model tracked in MLflow. The best model of each task is used by the app."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classification")
        st.table(pd.DataFrame(metrics["classification"]).T.round(4))
        cm = np.array(metrics["confusion_matrix"])
        show_in(
            st,
            px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                x=["Not good", "Good"],
                y=["Not good", "Good"],
                labels=dict(x="Predicted", y="Actual"),
                title=f"Confusion matrix - {metrics['best_clf']}",
            ).update_layout(height=380),
        )
    with c2:
        st.subheader("Regression")
        st.table(pd.DataFrame(metrics["regression"]).T.round(4))
        st.caption("RMSE / MAE are in ₹ Lakhs.")

    clf_imp, reg_imp = load_importance()
    i1, i2 = st.columns(2)
    show_in(
        i1,
        px.bar(
            clf_imp.head(15).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            title="Classifier feature importance",
        ).update_layout(height=480),
    )
    show_in(
        i2,
        px.bar(
            reg_imp.head(15).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            title="Regressor feature importance",
        ).update_layout(height=480),
    )

    st.info(
        "Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse every run, parameter and registered model."
    )


if page.startswith("🔮"):
    page_predict()
elif page.startswith("📊"):
    page_explorer()
else:
    page_insights()
