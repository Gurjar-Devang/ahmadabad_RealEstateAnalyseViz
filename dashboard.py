from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Ahmedabad Real Estate Dashboard",
    page_icon="AHM",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path(__file__).with_name("ahm_data.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("price_in_cr", "price")
        .str.replace("area_type", "area")
    )

    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", ""), errors="coerce")
    df["rate_per_sqft"] = pd.to_numeric(
        df["rate_per_sqft"].astype(str).str.replace(",", ""),
        errors="coerce",
    )
    df["area_in_sqft"] = pd.to_numeric(df["area_in_sqft"], errors="coerce")
    df["bhk_type"] = pd.to_numeric(df["bhk_type"], errors="coerce")

    df = df.dropna(how="all")
    df = df.dropna(subset=["price", "bhk_type", "area_in_sqft"])
    df = df[df["price"] > 0]
    df = df[df["area_in_sqft"] >= 100]
    df["rate_per_sqft"] = df["rate_per_sqft"].fillna(df["price"] * 10000000 / df["area_in_sqft"])
    df = df[df["rate_per_sqft"] < 15000]

    df["location"] = df["location"].fillna("Unknown").str.split(",").str[0].str.strip()
    df["property_type"] = df["property_type"].fillna("Unknown").str.title()
    df["area"] = df["area"].fillna("Unknown").str.title()
    df["property_title"] = df["property_title"].fillna(df["name"])
    df["bhk_label"] = df["bhk_type"].astype(int).astype(str) + " BHK"
    df["price_lakh"] = df["price"] * 100

    return df.reset_index(drop=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(199, 227, 255, 0.65), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 228, 196, 0.55), transparent 26%),
                linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f2742 0%, #1f4e79 100%);
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            color: #ffffff;
            box-shadow: 0 16px 40px rgba(15, 39, 66, 0.18);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            font-size: 2.1rem;
            margin: 0 0 0.35rem 0;
        }
        .hero-card p {
            margin: 0;
            opacity: 0.9;
            font-size: 1rem;
        }
        .kpi-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(31, 78, 121, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(36, 56, 99, 0.08);
            min-height: 135px;
        }
        .kpi-label {
            color: #5e6b7a;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            color: #102a43;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .kpi-note {
            color: #7b8794;
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }
        .section-title {
            color: #102a43;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.4rem 0 0.9rem 0;
        }
        .chart-note {
            background: rgba(255, 255, 255, 0.9);
            border-left: 4px solid #1f4e79;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-top: 0.35rem;
            color: #243b53;
            font-size: 0.92rem;
            line-height: 1.45;
            box-shadow: 0 8px 20px rgba(36, 56, 99, 0.08);
        }
        .chart-note strong {
            color: #102a43;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4f8fc 100%);
            border-right: 1px solid rgba(31, 78, 121, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_price_cr(value: float) -> str:
    return f"{value:.2f} Cr"


def format_currency(value: float) -> str:
    return f"Rs {value:,.0f}"


def score_to_label(score: float) -> str:
    if score >= 75:
        return "Hot Market"
    if score >= 55:
        return "Growth"
    if score >= 40:
        return "Balanced"
    return "Value Buy"


def normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        return pd.Series(50, index=series.index, dtype=float)
    scaled = (series - minimum) / (maximum - minimum) * 100
    return 100 - scaled if reverse else scaled


def build_location_kpi_model(filtered_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    location_perf = (
        filtered_df.groupby("location")
        .agg(
            listings=("location", "size"),
            avg_price_cr=("price", "mean"),
            avg_rate_sqft=("rate_per_sqft", "mean"),
            median_area_sqft=("area_in_sqft", "median"),
        )
        .reset_index()
    )

    if location_perf.empty:
        return location_perf

    ref_group = (
        reference_df.groupby("location")
        .agg(
            ref_listings=("location", "size"),
            ref_rate_sqft=("rate_per_sqft", "mean"),
            ref_area_sqft=("area_in_sqft", "median"),
            ref_price_cr=("price", "mean"),
        )
        .reset_index()
    )

    location_perf = location_perf.merge(ref_group, on="location", how="left")
    location_perf["inventory_score"] = normalize(location_perf["ref_listings"])
    location_perf["value_score"] = normalize(location_perf["ref_rate_sqft"], reverse=True)
    location_perf["space_score"] = normalize(location_perf["ref_area_sqft"])
    location_perf["premium_score"] = normalize(location_perf["ref_price_cr"])
    location_perf["market_score"] = (
        location_perf["inventory_score"] * 0.35
        + location_perf["value_score"] * 0.30
        + location_perf["space_score"] * 0.20
        + location_perf["premium_score"] * 0.15
    ).round(1)
    location_perf["signal"] = location_perf["market_score"].apply(score_to_label)

    return location_perf.sort_values(["market_score", "listings"], ascending=[False, False])


def create_kpi_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def explain_chart(title: str, insight: str, x_axis: str, y_axis: str) -> None:
    st.markdown(
        f"""
        <div class="chart-note">
            <strong>{title}</strong><br>
            {insight}<br>
            <strong>X-axis:</strong> {x_axis}<br>
            <strong>Y-axis:</strong> {y_axis}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_custom_bar_data(
    source_df: pd.DataFrame,
    x_field: str,
    y_field: str,
    aggregation: str,
) -> pd.DataFrame:
    grouped = (
        source_df.groupby(x_field)
        .agg(metric=(y_field, aggregation), listings=("location", "size"))
        .reset_index()
        .sort_values("metric", ascending=False)
        .head(12)
    )
    return grouped


inject_styles()
df = load_data()

st.markdown(
    """
    <div class="hero-card">
        <h1>Ahmedabad Real Estate Intelligence Dashboard</h1>
        <p>Professional market view with executive KPIs, interactive filters, and a location scoring model for faster decision-making.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Filters")
selected_locations = st.sidebar.multiselect(
    "Choose Location Name",
    options=sorted(df["location"].unique()),
    placeholder="Choose Location Name",
)
selected_bhk = st.sidebar.multiselect(
    "Choose BHK Type",
    options=sorted(df["bhk_label"].unique(), key=lambda x: int(x.split()[0])),
    placeholder="Choose BHK Type",
)
selected_property_types = st.sidebar.multiselect(
    "Choose Property Type",
    options=sorted(df["property_type"].unique()),
    placeholder="Choose Property Type",
)
selected_area_types = st.sidebar.multiselect(
    "Choose Area Category",
    options=sorted(df["area"].unique()),
    placeholder="Choose Area Category",
)
price_range = st.sidebar.slider(
    "Choose Price Range (Cr)",
    min_value=float(df["price"].min()),
    max_value=float(df["price"].quantile(0.99)),
    value=(float(df["price"].min()), float(df["price"].quantile(0.90))),
)

filtered_df = df.copy()
if selected_locations:
    filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]
if selected_bhk:
    filtered_df = filtered_df[filtered_df["bhk_label"].isin(selected_bhk)]
if selected_property_types:
    filtered_df = filtered_df[filtered_df["property_type"].isin(selected_property_types)]
if selected_area_types:
    filtered_df = filtered_df[filtered_df["area"].isin(selected_area_types)]

filtered_df = filtered_df[
    filtered_df["price"].between(price_range[0], price_range[1], inclusive="both")
]

if filtered_df.empty:
    st.warning("No listings match the current filters. Try widening the selection.")
    st.stop()

top_location = (
    filtered_df.groupby("location")["price"].mean().sort_values(ascending=False).index[0]
)
costliest_listing = filtered_df.loc[filtered_df["price"].idxmax()]
cheapest_listing = filtered_df.loc[filtered_df["price"].idxmin()]
location_kpi_model = build_location_kpi_model(filtered_df, df)
best_scored_market = location_kpi_model.iloc[0] if not location_kpi_model.empty else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    create_kpi_card(
        "Live Listings",
        f"{len(filtered_df):,}",
        f"{filtered_df['location'].nunique()} locations in view",
    )
with col2:
    create_kpi_card(
        "Average Ticket Size",
        format_price_cr(filtered_df["price"].mean()),
        f"Median: {format_price_cr(filtered_df['price'].median())}",
    )
with col3:
    create_kpi_card(
        "Median Rate / Sqft",
        format_currency(filtered_df["rate_per_sqft"].median()),
        f"Average: {format_currency(filtered_df['rate_per_sqft'].mean())}",
    )
with col4:
    create_kpi_card(
        "Prime Location",
        top_location,
        f"Highest average price among filtered markets",
    )

st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
summary_col1, summary_col2 = st.columns([1.3, 1])

with summary_col1:
    scatter_fig = px.scatter(
        filtered_df,
        x="area_in_sqft",
        y="price",
        color="bhk_label",
        size="rate_per_sqft",
        hover_data=["location", "property_title", "property_type"],
        title="Price vs Area by BHK Mix",
        labels={"area_in_sqft": "Area (sqft)", "price": "Price (Cr)", "bhk_label": "BHK"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    scatter_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        legend_title_text="BHK",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#102a43"),
        xaxis_title="Area (sqft)",
        yaxis_title="Price (Cr)",
    )
    st.plotly_chart(scatter_fig, width="stretch")
    explain_chart(
        "Price vs Area by BHK Mix",
        "Yeh chart dikhata hai ki alag-alag BHK listings me area badhne par price kaise change hota hai",
        "Property area in square feet",
        "Property price in crore",
    )

with summary_col2:
    donut_fig = px.pie(
        filtered_df,
        names="property_type",
        hole=0.62,
        title="Inventory Mix",
        color_discrete_sequence=["#1f4e79", "#f4a261", "#2a9d8f", "#6d597a"],
    )
    donut_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#102a43"),
    )
    st.plotly_chart(donut_fig, width="stretch")
    st.markdown(
        """
        <div class="chart-note">
            <strong>Inventory Mix</strong><br>
            Yeh chart batata hai ki filtered listings me har property type ka kitna share hai.
        </div>
        """,
        unsafe_allow_html=True,
    )

mid_col1, mid_col2 = st.columns(2)

with mid_col1:
    top_locations = (
        filtered_df.groupby("location")
        .agg(avg_price=("price", "mean"), listings=("location", "size"))
        .sort_values("avg_price", ascending=False)
        .head(10)
        .reset_index()
    )
    bar_fig = px.bar(
        top_locations,
        x="avg_price",
        y="location",
        orientation="h",
        color="listings",
        title="Top 10 Locations by Average Price",
        labels={"avg_price": "Average Price (Cr)", "location": "", "listings": "Listings"},
        color_continuous_scale="Blues",
    )
    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#102a43"),
        xaxis_title="Average Price (Cr)",
        yaxis_title="Location Name",
    )
    st.plotly_chart(bar_fig, width="stretch")
    explain_chart(
        "Top 10 Locations by Average Price",
        "Yeh chart sabse mehengi average price wale top locations ko compare karta hai",
        "Average property price in crore",
        "Location name",
    )

with mid_col2:
    bhk_trend = (
        filtered_df.groupby("bhk_label")
        .agg(avg_price=("price", "mean"), avg_rate=("rate_per_sqft", "mean"))
        .reset_index()
    )
    bhk_trend["sort_key"] = bhk_trend["bhk_label"].str.extract(r"(\d+)").astype(int)
    bhk_trend = bhk_trend.sort_values("sort_key")
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=bhk_trend["bhk_label"],
            y=bhk_trend["avg_price"],
            mode="lines+markers",
            name="Avg Price (Cr)",
            line=dict(color="#1f4e79", width=3),
        )
    )
    trend_fig.add_trace(
        go.Scatter(
            x=bhk_trend["bhk_label"],
            y=bhk_trend["avg_rate"] / 10000,
            mode="lines+markers",
            name="Avg Rate / Sqft (x10k)",
            line=dict(color="#f4a261", width=3),
            yaxis="y2",
        )
    )
    trend_fig.update_layout(
        title="BHK Pricing Trend",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis=dict(title="Price (Cr)"),
        yaxis2=dict(title="Rate / Sqft", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(color="#102a43"),
        xaxis_title="BHK Type",
    )
    st.plotly_chart(trend_fig, width="stretch")
    st.markdown(
        """
        <div class="chart-note">
            <strong>BHK Pricing Trend</strong><br>
            Yeh chart batata hai ki har BHK type me average price aur average rate per sqft ka trend kya hai.<br>
            <strong>X-axis:</strong> BHK type<br>
            <strong>Left Y-axis:</strong> Average price in crore<br>
            <strong>Right Y-axis:</strong> Average rate per sqft
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Custom Bar Chart</div>', unsafe_allow_html=True)
chart_control_col1, chart_control_col2, chart_control_col3, chart_control_col4 = st.columns(4)

chart_field_labels = {
    "location": "Location Name",
    "bhk_label": "BHK Type",
    "property_type": "Property Type",
    "area": "Area Category",
}
y_field_labels = {
    "price": "Price (Cr)",
    "rate_per_sqft": "Rate per Sqft",
    "area_in_sqft": "Area in Sqft",
}

with chart_control_col1:
    custom_chart_title = st.text_input("Chart Name", value="Location Wise Price Analysis")
with chart_control_col2:
    chart_x_field = st.selectbox(
        "Choose X Axis",
        options=list(chart_field_labels.keys()),
        format_func=lambda x: chart_field_labels[x],
    )
with chart_control_col3:
    chart_y_field = st.selectbox(
        "Choose Y Axis",
        options=list(y_field_labels.keys()),
        format_func=lambda y: y_field_labels[y],
    )
with chart_control_col4:
    chart_aggregation = st.selectbox(
        "Y Value Type",
        options=["mean", "median", "max", "min"],
        format_func=lambda a: a.title(),
    )

custom_bar_data = build_custom_bar_data(filtered_df, chart_x_field, chart_y_field, chart_aggregation)
custom_bar_fig = px.bar(
    custom_bar_data,
    x=chart_x_field,
    y="metric",
    color="listings",
    text_auto=".2s",
    title=custom_chart_title,
    labels={
        chart_x_field: chart_field_labels[chart_x_field],
        "metric": f"{chart_aggregation.title()} {y_field_labels[chart_y_field]}",
        "listings": "Listings",
    },
    color_continuous_scale="Tealgrn",
)
custom_bar_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.85)",
    margin=dict(l=20, r=20, t=55, b=20),
    font=dict(color="#102a43"),
    xaxis_title=chart_field_labels[chart_x_field],
    yaxis_title=f"{chart_aggregation.title()} {y_field_labels[chart_y_field]}",
)
st.plotly_chart(custom_bar_fig, width="stretch")
explain_chart(
    custom_chart_title,
    f"Yeh custom chart selected category ke against {chart_aggregation} {y_field_labels[chart_y_field].lower()} dikhata hai",
    chart_field_labels[chart_x_field],
    f"{chart_aggregation.title()} {y_field_labels[chart_y_field]}",
)

st.markdown('<div class="section-title">KPI Model</div>', unsafe_allow_html=True)
model_col1, model_col2 = st.columns([1.15, 0.85])

with model_col1:
    if best_scored_market is not None:
        st.info(
            f"Best scored location right now: {best_scored_market['location']} "
            f"with a market score of {best_scored_market['market_score']:.1f}/100 "
            f"and signal '{best_scored_market['signal']}'."
        )

    display_model = location_kpi_model[
        [
            "location",
            "market_score",
            "signal",
            "listings",
            "avg_price_cr",
            "avg_rate_sqft",
            "median_area_sqft",
        ]
    ].head(12).copy()
    display_model.columns = [
        "Location",
        "Market Score",
        "Signal",
        "Listings",
        "Avg Price (Cr)",
        "Avg Rate / Sqft",
        "Median Area (sqft)",
    ]
    st.dataframe(display_model, width="stretch", hide_index=True)

with model_col2:
    score_fig = px.bar(
        location_kpi_model.head(8),
        x="market_score",
        y="location",
        orientation="h",
        color="signal",
        title="Top Markets by KPI Score",
        labels={"market_score": "KPI Score", "location": ""},
        color_discrete_map={
            "Hot Market": "#d62828",
            "Growth": "#f77f00",
            "Balanced": "#1d3557",
            "Value Buy": "#2a9d8f",
        },
    )
    score_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#102a43"),
        xaxis_title="KPI Score",
        yaxis_title="Location Name",
    )
    st.plotly_chart(score_fig, width="stretch")
    explain_chart(
        "Top Markets by KPI Score",
        "Yeh chart highest market score wale locations ko rank karta hai",
        "KPI score",
        "Location name",
    )

bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
with bottom_col1:
    create_kpi_card(
        "Costliest Listing",
        format_price_cr(costliest_listing["price"]),
        f"{costliest_listing['property_title']} | {costliest_listing['location']}",
    )
with bottom_col2:
    create_kpi_card(
        "Most Affordable Listing",
        format_price_cr(cheapest_listing["price"]),
        f"{cheapest_listing['property_title']} | {cheapest_listing['location']}",
    )
with bottom_col3:
    create_kpi_card(
        "Average Home Size",
        f"{filtered_df['area_in_sqft'].median():,.0f} sqft",
        f"Typical listing size across selected inventory",
    )

st.caption(
    "KPI model uses weighted inventory depth, price value, space efficiency, and premium positioning to score locations."
)
