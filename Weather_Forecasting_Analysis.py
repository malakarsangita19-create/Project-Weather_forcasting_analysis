import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Weather Data Analysis & Forecasting",
    layout="wide",
    page_icon="🌦️"
)

# =====================================================
# LIGHT PURPLE PROFESSIONAL THEME
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F3E8FF;
    color: #1F2937;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E9D5FF;
}

/* Header container */
.header-box {
    background: linear-gradient(90deg, #A855F7, #7C3AED);
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Remove plot white block effect */
.js-plotly-plot .plotly .main-svg {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class='header-box'>
<h1>🌦️ Weather Data Analysis & Forecasting</h1>
<p>Global Climate Intelligence | Heatmaps | AI Forecasting | Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_geo():
    cities = pd.read_csv(r"C:\data_science_project\cities (1).csv")
    countries = pd.read_csv(r"C:\data_science_project\countries.csv")

    cities.columns = cities.columns.str.lower().str.strip()
    countries.columns = countries.columns.str.lower().str.strip()

    if "latitude" in cities.columns:
        cities.rename(columns={"latitude": "lat"}, inplace=True)
    if "longitude" in cities.columns:
        cities.rename(columns={"longitude": "lng"}, inplace=True)

    cities["city_name"] = cities["city_name"].astype(str)
    cities = cities.dropna(subset=["city_name"])

    geo = cities.merge(
        countries[["iso3", "continent"]],
        on="iso3",
        how="left"
    )

    return geo


@st.cache_data
def load_weather():
    df = pd.read_parquet(
        r"C:\data_science_project\daily_weather.parquet",
        columns=["date", "avg_temp_c", "city_name"]
    )

    df.columns = df.columns.str.lower().str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["avg_temp_c"] = pd.to_numeric(df["avg_temp_c"], errors="coerce")
    df["city_name"] = df["city_name"].astype(str)

    df = df.dropna()

    return df


geo_df = load_geo()
weather = load_weather()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🌍 Filters")

continents = sorted(geo_df["continent"].dropna().unique())
selected_continent = st.sidebar.selectbox("Select Continent", continents)

cities_list = geo_df[geo_df["continent"] == selected_continent]["city_name"].dropna().unique()
selected_city = st.sidebar.selectbox("Select City", sorted(cities_list))

unit = st.sidebar.radio("Temperature Unit", ["Celsius (°C)", "Fahrenheit (°F)"])

# =====================================================
# FILTER DATA
# =====================================================

filtered = weather[weather["city_name"] == selected_city].copy()
filtered = filtered.sort_values("date")

if unit == "Fahrenheit (°F)":
    filtered["temperature"] = filtered["avg_temp_c"] * 9/5 + 32
else:
    filtered["temperature"] = filtered["avg_temp_c"]

# =====================================================
# KPI SECTION
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric("🌡 Average", round(filtered["temperature"].mean(), 2))
col2.metric("🔥 Max", round(filtered["temperature"].max(), 2))
col3.metric("❄ Min", round(filtered["temperature"].min(), 2))

st.markdown("---")

st.markdown(f"**Selected Continent:** {selected_continent} | **Selected City:** {selected_city}")

# =====================================================
# DAILY TREND
# =====================================================

fig_daily = px.line(
    filtered,
    x="date",
    y="temperature",
    template="plotly_white",
    title="Daily Temperature Trend"
)

fig_daily.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_daily, use_container_width=True)

trend_direction = "increasing" if filtered["temperature"].iloc[-1] > filtered["temperature"].iloc[0] else "decreasing"

st.info(f"""
📌 **Conclusion:**  
The overall temperature trend in {selected_city} ({selected_continent}) appears to be **{trend_direction}** over time.  
This suggests long-term climate variation patterns in the selected region.
""")

# =====================================================
# GLOBAL CHOROPLETH
# =====================================================

st.subheader("🌍 Global Temperature Choropleth")

map_df = weather.merge(
    geo_df[["city_name", "iso3", "continent"]],
    on="city_name",
    how="left"
)

country_avg = map_df.groupby("iso3")["avg_temp_c"].mean().reset_index()

if unit == "Fahrenheit (°F)":
    country_avg["temperature"] = country_avg["avg_temp_c"] * 9/5 + 32
else:
    country_avg["temperature"] = country_avg["avg_temp_c"]

fig_choropleth = px.choropleth(
    country_avg,
    locations="iso3",
    color="temperature",
    color_continuous_scale="Turbo",
    title="Average Temperature by Country"
)

fig_choropleth.update_layout(
    template="plotly_white",
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_choropleth, use_container_width=True)

st.info("""
📌 **Conclusion:**  
The global heatmap highlights temperature distribution differences across countries,  
showing how climate varies geographically worldwide.
""")

# =====================================================
# ANIMATED MAP
# =====================================================

st.subheader("🎬 Animated Climate Change Over Years")

weather["year"] = weather["date"].dt.year

year_avg = weather.merge(
    geo_df[["city_name", "iso3"]],
    on="city_name",
    how="left"
)

year_avg = year_avg.groupby(["year", "iso3"])["avg_temp_c"].mean().reset_index()

fig_anim = px.choropleth(
    year_avg,
    locations="iso3",
    color="avg_temp_c",
    animation_frame="year",
    color_continuous_scale="Turbo",
    title="Global Climate Change Over Years"
)

fig_anim.update_layout(
    template="plotly_white",
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_anim, use_container_width=True)

st.info("""
📌 **Conclusion:**  
The animation demonstrates long-term global warming trends,  
indicating gradual temperature rise across multiple regions over decades.
""")

# =====================================================
# ANOMALY DETECTION
# =====================================================

st.subheader("🚨 Trend Anomaly Detection")

filtered["year_month"] = filtered["date"].dt.to_period("M")
monthly = filtered.groupby("year_month")["temperature"].mean().reset_index()
monthly["year_month"] = monthly["year_month"].dt.to_timestamp()

monthly["rolling"] = monthly["temperature"].rolling(12).mean()
monthly["deviation"] = monthly["temperature"] - monthly["rolling"]

threshold = monthly["deviation"].std() * 2
anomalies = monthly[np.abs(monthly["deviation"]) > threshold]

fig_anomaly = px.line(
    monthly,
    x="year_month",
    y="temperature",
    template="plotly_white",
    title="Temperature Anomaly Detection"
)

fig_anomaly.add_scatter(
    x=anomalies["year_month"],
    y=anomalies["temperature"],
    mode="markers",
    marker=dict(color="red", size=8),
    name="Anomaly"
)

fig_anomaly.update_layout(
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_anomaly, use_container_width=True)

st.success(f"{len(anomalies)} climate anomalies detected.")

st.info("""
📌 **Conclusion:**  
Detected anomalies represent unusual temperature deviations  
which may indicate extreme weather events or climate shifts.
""")

# =====================================================
# AI FORECASTING
# =====================================================

st.subheader("📈 AI Forecasting (Prophet)")

prophet_df = filtered[["date", "temperature"]].rename(
    columns={"date": "ds", "temperature": "y"}
)

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig_forecast = px.line(
    forecast,
    x="ds",
    y="yhat",
    template="plotly_white",
    title="1-Year AI Temperature Forecast"
)

fig_forecast.update_layout(
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.info(f"""
📌 **Conclusion:**  
The AI model forecasts future temperature trends for {selected_city}.  
This predictive analysis helps anticipate possible climate shifts over the next year.
""")

# =====================================================
# 🔥❄ EXTREME EVENTS TRACKER (ADDED SECTION)
# =====================================================

st.subheader("🔥❄ Extreme Events Tracker")

heat_threshold = np.percentile(filtered["temperature"], 95)
cold_threshold = np.percentile(filtered["temperature"], 5)

heatwave_days = filtered[filtered["temperature"] > heat_threshold]
coldwave_days = filtered[filtered["temperature"] < cold_threshold]

col1, col2 = st.columns(2)

col1.metric("🔥 Heatwave Days (Above 95th Percentile)", len(heatwave_days))
col2.metric("❄ Cold Wave Days (Below 5th Percentile)", len(coldwave_days))

st.info(f"""
📌 **Extreme Events Insight:**  
• Heatwave threshold: {round(heat_threshold,2)}  
• Cold wave threshold: {round(cold_threshold,2)}  

These represent statistically extreme temperature events in {selected_city}.
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<hr>
<center>
© 2026 Weather Data Analysis & Forecasting | Developed by Sangita & Rakhi
</center>
""", unsafe_allow_html=True)
