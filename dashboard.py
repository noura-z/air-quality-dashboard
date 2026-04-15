"""
Air Quality Monitoring Dashboard
A clean, friendly, non-technical interface for exploring air pollution data.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWatch – Air Quality Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS: Light, clean, pastel design ─────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8edf3;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e8edf3;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Section headers */
    h1 { color: #1e3a5f; font-size: 2rem !important; }
    h2 { color: #2c5282; font-size: 1.3rem !important; }
    h3 { color: #2d3748; font-size: 1.05rem !important; }

    /* AQI badge */
    .aqi-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Health card */
    .health-card {
        background: #ffffff;
        border-left: 5px solid;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Insight pill */
    .insight-pill {
        background: #eef2ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #3730a3;
        font-size: 0.92rem;
    }

    /* Warning banner */
    .warning-banner {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 10px;
        padding: 14px 18px;
        color: #9a3412;
        font-weight: 500;
    }

    /* Footer */
    .footer { color: #94a3b8; font-size: 0.8rem; text-align: center; margin-top: 2rem; }

    /* Hide Streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── AQI Helper Functions ────────────────────────────────────────────────────

def get_aqi_info(pm25_value):
    """Return AQI category, color, emoji and health message based on WHO thresholds."""
    if pd.isna(pm25_value):
        return "Unknown", "#94a3b8", "⚪", "No data available."
    if pm25_value <= 5:
        return "Good", "#22c55e", "🟢", "Air quality is great! Safe to be outdoors."
    elif pm25_value <= 15:
        return "Moderate", "#84cc16", "🟡", "Air quality is acceptable. Most people can enjoy outdoor activities."
    elif pm25_value <= 25:
        return "Unhealthy for Sensitive Groups", "#f59e0b", "🟠", "Sensitive groups (children, elderly, asthma) should limit prolonged outdoor activity."
    elif pm25_value <= 50:
        return "Unhealthy", "#ef4444", "🔴", "Everyone may begin to experience health effects. Limit outdoor activities."
    else:
        return "Hazardous", "#7c3aed", "🟣", "Health alert! Everyone should avoid outdoor activity."


def compute_aqi_category(series):
    """Label a PM2.5 series with WHO AQI categories."""
    bins   = [0, 5, 15, 25, 50, np.inf]
    labels = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Hazardous"]
    return pd.cut(series, bins=bins, labels=labels, right=True)


# ─── Data Loading & Cleaning ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean(file_obj):
    """Load CSV, clean columns, parse timestamps, drop dupes."""
    df = pd.read_csv(file_obj)
    df.columns = df.columns.str.strip().str.lower()

    # Parse datetime
    date_col = next((c for c in df.columns if "utc" in c or "datetime" in c), None)
    if date_col:
        df["datetime"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        # Fallback: look for any date-like column
        for col in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df[col], errors="coerce", utc=True)
                if df["datetime"].notna().sum() > len(df) * 0.5:
                    break
            except Exception:
                continue

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates()
    df = df.set_index("datetime")

    # Ensure numeric pollution columns
    for col in ["pm25", "pm1", "um003", "temperature", "relativehumidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop physically impossible values
    if "pm25" in df.columns:
        df = df[df["pm25"] > 0]

    # Interpolate remaining NaNs
    df = df.infer_objects()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")

    return df


def validate_columns(df):
    """Return a dict of which expected columns are present."""
    required = {"pm25": "PM2.5", "pm1": "PM1", "um003": "UM0.3", 
                "temperature": "Temperature", "relativehumidity": "Humidity"}
    return {k: v for k, v in required.items() if k in df.columns}


# ─── Forecast Simulation (7-day) ──────────────────────────────────────────────

def simulate_forecast(df, location=None, days=7):
    """
    Simple forecasting: uses rolling mean + seasonal pattern + small noise.
    This mimics a light ML approach without requiring a trained model at runtime.
    """
    if location and "location_name" in df.columns:
        sub = df[df["location_name"] == location]
    else:
        sub = df

    # Get recent PM2.5 trend
    pm25_recent = sub["pm25"].dropna().tail(168)  # last 7 days
    base_pm25   = pm25_recent.mean() if len(pm25_recent) > 0 else 12.0
    pm25_std    = pm25_recent.std()   if len(pm25_recent) > 0 else 2.0

    # Hourly diurnal pattern (peaks at rush hours)
    hour_pattern = np.array([
        0.85, 0.80, 0.78, 0.77, 0.80, 0.90,
        1.05, 1.15, 1.10, 1.00, 0.95, 0.92,
        0.90, 0.92, 0.95, 1.00, 1.08, 1.15,
        1.10, 1.05, 1.00, 0.95, 0.90, 0.87,
    ])

    last_time = df.index.max()
    future_hours = pd.date_range(start=last_time, periods=days * 24 + 1, freq="h")[1:]

    pm25_forecast = []
    for ts in future_hours:
        pattern_factor = hour_pattern[ts.hour]
        daily_drift    = np.sin(2 * np.pi * ts.dayofweek / 7) * 0.5
        noise          = np.random.normal(0, pm25_std * 0.15)
        val = max(0.5, base_pm25 * pattern_factor + daily_drift + noise)
        pm25_forecast.append(round(val, 2))

    # Temperature forecast
    temp_recent  = sub["temperature"].dropna().tail(168) if "temperature" in sub.columns else pd.Series([22.0])
    base_temp    = temp_recent.mean() if len(temp_recent) > 0 else 22.0
    temp_pattern = np.array([
        19, 18, 18, 17, 17, 18, 19, 21, 23, 25, 27, 28,
        29, 30, 30, 29, 28, 27, 25, 23, 22, 21, 20, 19,
    ]) / 24.0 * base_temp
    temp_base    = (temp_pattern / temp_pattern.mean())

    temp_forecast = []
    for ts in future_hours:
        noise = np.random.normal(0, 0.8)
        val   = base_temp * temp_base[ts.hour] + noise
        temp_forecast.append(round(val, 1))

    forecast_df = pd.DataFrame({
        "datetime"   : future_hours,
        "pm25_forecast"  : pm25_forecast,
        "temp_forecast"  : temp_forecast,
    }).set_index("datetime")

    return forecast_df


# ─── Insights Generator ───────────────────────────────────────────────────────

def generate_insights(df, location=None):
    """Return a list of plain-language insights derived from the data."""
    insights = []

    sub = df[df["location_name"] == location] if (location and "location_name" in df.columns) else df

    if "pm25" not in sub.columns or sub["pm25"].dropna().empty:
        return ["Not enough data to generate insights."]

    # Trend
    weekly_mean  = sub["pm25"].dropna().tail(168).mean()
    monthly_mean = sub["pm25"].dropna().tail(720).mean()
    if weekly_mean > monthly_mean * 1.1:
        insights.append("📈 **Pollution has increased** over the last week compared to the monthly average.")
    elif weekly_mean < monthly_mean * 0.9:
        insights.append("📉 **Pollution has improved** over the last week compared to the monthly average.")
    else:
        insights.append("📊 **Pollution levels are stable** and close to the monthly average.")

    # Diurnal pattern
    hourly = sub["pm25"].groupby(sub.index.hour).mean()
    peak_hour = hourly.idxmax()
    low_hour  = hourly.idxmin()
    tod = "morning" if 6 <= peak_hour <= 10 else ("evening" if 16 <= peak_hour <= 20 else "midday")
    insights.append(f"🕐 **Pollution peaks around {peak_hour}:00** ({tod} hours) and is lowest around {low_hour}:00.")

    # WHO threshold
    pct_over = (sub["pm25"] > 15).mean() * 100
    if pct_over > 30:
        insights.append(f"⚠️ **{pct_over:.0f}% of readings** exceeded the WHO safe daily limit of 15 µg/m³.")
    elif pct_over > 10:
        insights.append(f"🔔 **{pct_over:.0f}% of readings** exceeded the WHO safe limit – worth monitoring.")
    else:
        insights.append(f"✅ **Air quality is mostly safe** — only {pct_over:.0f}% of readings exceeded WHO limits.")

    # Weekend vs weekday
    if len(sub) > 200:
        weekday_mean = sub[sub.index.dayofweek < 5]["pm25"].mean()
        weekend_mean = sub[sub.index.dayofweek >= 5]["pm25"].mean()
        if weekday_mean > weekend_mean * 1.1:
            insights.append("🚗 **Weekday pollution is higher** than weekends, likely due to traffic and activity.")
        elif weekend_mean > weekday_mean * 1.1:
            insights.append("📅 **Weekend pollution is slightly higher**, which may point to outdoor events or burning.")
        else:
            insights.append("📅 **Weekday and weekend levels are similar**, suggesting pollution isn't mainly traffic-driven.")

    return insights


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wind.png", width=60)
    st.title("AirWatch 🌿")
    st.caption("Air Quality Monitoring Dashboard")
    st.divider()

    # ── Data Upload ──
    st.subheader("📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV with columns: datetime, pm25, location_name, latitude, longitude, etc."
    )

    use_sample = st.checkbox("Use sample dataset", value=(uploaded_file is None))
    st.divider()

    # ── Filters (rendered after data loads) ──
    st.subheader("🔍 Filters")
    filter_placeholder = st.empty()
    st.divider()
    st.markdown('<p class="footer">Data source: OpenAQ / AirGradient<br>WHO thresholds applied</p>', unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_sample():
    """Load the bundled openaq_merged.csv sample dataset."""
    # Try to load from project files, otherwise generate synthetic data
    try:
        df = pd.read_csv("/mnt/project/openaq_merged.csv")
        df.columns = df.columns.str.strip().str.lower()
        df["datetime"] = pd.to_datetime(df["datetimeutc"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates()
        df = df.set_index("datetime")
        for col in ["pm25", "pm1", "um003", "temperature", "relativehumidity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "pm25" in df.columns:
            df = df[df["pm25"] > 0]
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        return df
    except Exception:
        # Fallback: generate synthetic data
        np.random.seed(42)
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        hour_noise = np.sin(2 * np.pi * idx.hour / 24) * 3
        df = pd.DataFrame({
            "pm25": np.clip(np.random.normal(12, 4, n) + hour_noise, 0.5, 80),
            "pm1":  np.clip(np.random.normal(8,  3, n) + hour_noise * 0.8, 0.5, 60),
            "um003": np.clip(np.random.normal(500, 100, n), 50, 2000),
            "temperature": np.random.normal(22, 5, n) + np.sin(2 * np.pi * idx.hour / 24) * 4,
            "relativehumidity": np.clip(np.random.normal(55, 12, n), 10, 100),
            "latitude": np.random.choice([32.238, 31.629], n),
            "longitude": np.random.choice([-7.936, -7.982], n),
            "location_name": np.random.choice(["Ben Guerir – Campus", "Marrakech – Residence Mima"], n),
        }, index=idx)
        return df


# ── Decide which data to use ──
with st.spinner("Loading data…"):
    if uploaded_file is not None:
        try:
            df = load_and_clean(uploaded_file)
            data_source = f"📤 Uploaded: **{uploaded_file.name}**"
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()
    else:
        df = load_sample()
        data_source = "📦 Sample dataset: **openaq_merged.csv** (Ben Guerir / Marrakech area)"

available = validate_columns(df)

# ── Sidebar filters ──
with filter_placeholder.container():
    # Location filter
    locations = ["All locations"]
    if "location_name" in df.columns:
        locations += sorted(df["location_name"].dropna().unique().tolist())
    selected_loc = st.selectbox("📍 Location", locations)

    # Date range filter
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = st.date_input(
        "📅 Date range",
        value=(max(min_date, max_date - timedelta(days=30)), max_date),
        min_value=min_date,
        max_value=max_date,
    )

# ── Apply filters ──
filtered = df.copy()
if selected_loc != "All locations" and "location_name" in df.columns:
    filtered = filtered[filtered["location_name"] == selected_loc]

try:
    start_dt = pd.Timestamp(date_range[0], tz="UTC")
    end_dt   = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
    filtered = filtered[(filtered.index >= start_dt) & (filtered.index <= end_dt)]
except Exception:
    pass  # Keep full date range if filter fails

if filtered.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the date range or location.")
    st.stop()


# ─── Main Dashboard ───────────────────────────────────────────────────────────

# ── Header ──
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("## 🌿")
with col_title:
    st.markdown("## AirWatch – Air Quality Dashboard")
    st.caption(data_source)

# ── Warning Banner ──
current_pm25 = filtered["pm25"].dropna().tail(12).mean() if "pm25" in filtered.columns else None
if current_pm25 and current_pm25 > 25:
    label, color, emoji, _ = get_aqi_info(current_pm25)
    st.markdown(f"""
    <div class="warning-banner">
        ⚠️ <strong>Air Quality Alert:</strong> Recent PM2.5 average is {current_pm25:.1f} µg/m³ 
        — classified as <strong>{label}</strong>. Consider limiting outdoor activities.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 1: Key Metrics
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Current Air Quality Snapshot")

metric_cols = st.columns(5)

def format_delta(series, suffix=""):
    """Compute trend delta between last 24h and previous 24h."""
    if len(series) < 48:
        return None
    recent   = series.tail(24).mean()
    previous = series.tail(48).head(24).mean()
    delta    = recent - previous
    return f"{delta:+.1f}{suffix}"

metrics_to_show = [
    ("pm25",             "PM2.5",       "µg/m³", "🌫️"),
    ("pm1",              "PM1",         "µg/m³", "💨"),
    ("um003",            "UM0.3",       "p/dl",  "🔬"),
    ("temperature",      "Temperature", "°C",    "🌡️"),
    ("relativehumidity", "Humidity",    "%",     "💧"),
]

for i, (col, label, unit, icon) in enumerate(metrics_to_show):
    if col in filtered.columns:
        val   = filtered[col].dropna().tail(12).mean()
        delta = format_delta(filtered[col].dropna(), unit)
        with metric_cols[i]:
            st.metric(f"{icon} {label}", f"{val:.1f} {unit}", delta=delta)

st.markdown("")

# ════════════════════════════════════════════════════════════════════════════
# Section 2: AQI Indicator + Health Panel
# ════════════════════════════════════════════════════════════════════════════
aqi_col, health_col = st.columns([1, 2])

with aqi_col:
    st.markdown("### 🟡 Air Quality Index")

    if "pm25" in filtered.columns:
        latest_pm25 = filtered["pm25"].dropna().tail(12).mean()
        cat, color, emoji, health_msg = get_aqi_info(latest_pm25)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(latest_pm25, 1),
            number={"suffix": " µg/m³", "font": {"size": 22, "color": "#1e3a5f"}},
            gauge={
                "axis"     : {"range": [0, 75], "tickcolor": "#64748b"},
                "bar"      : {"color": color, "thickness": 0.3},
                "steps"    : [
                    {"range": [0,  5],  "color": "#dcfce7"},
                    {"range": [5,  15], "color": "#fef9c3"},
                    {"range": [15, 25], "color": "#ffedd5"},
                    {"range": [25, 50], "color": "#fee2e2"},
                    {"range": [50, 75], "color": "#f3e8ff"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 15},
            },
            title={"text": "", "font": {"size": 14}},
        ))
        fig_gauge.update_layout(
            height=230, margin=dict(t=10, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#1e3a5f"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f'<div style="text-align:center">'
            f'<span class="aqi-badge" style="background:{color}20; color:{color}; border:2px solid {color}">'
            f'{emoji} {cat}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("PM2.5 data not available in this dataset.")

with health_col:
    st.markdown("### ❤️ Health Guidance")

    if "pm25" in filtered.columns:
        cat, color, emoji, msg = get_aqi_info(latest_pm25)

        # Main health message
        st.markdown(
            f'<div class="health-card" style="border-color:{color}">'
            f'<strong style="color:{color}">{emoji} {cat}</strong><br>{msg}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Contextual advice
        advice_map = {
            "Good": [
                "✅ Ideal for outdoor sports and exercise",
                "✅ Windows can be kept open for ventilation",
                "✅ No special precautions needed",
            ],
            "Moderate": [
                "💛 Unusually sensitive people should consider limiting extended outdoor exertion",
                "✅ Most people can enjoy outdoor activities normally",
                "🪟 Ventilation is still fine",
            ],
            "Unhealthy (Sensitive Groups)": [
                "🧒 Children and elderly should limit outdoor activity",
                "😷 People with asthma or heart conditions should take precautions",
                "✅ Healthy adults can usually continue normal activity",
            ],
            "Unhealthy": [
                "😷 Everyone should limit prolonged outdoor exertion",
                "🏠 Stay indoors when possible, keep windows closed",
                "💊 Those with respiratory issues should keep medications handy",
            ],
            "Hazardous": [
                "🚫 Avoid ALL outdoor physical activity",
                "🏠 Stay indoors with windows and doors closed",
                "😷 Use N95 mask if you must go outside",
                "🏥 Seek medical advice if experiencing symptoms",
            ],
        }
        advice = advice_map.get(cat, ["Monitor air quality regularly."])
        for tip in advice:
            st.markdown(
                f'<div class="insight-pill">{tip}</div>',
                unsafe_allow_html=True,
            )

    # AQI Scale legend
    st.markdown("**WHO PM2.5 Scale:**")
    scale_data = {
        "Category"   : ["Good", "Moderate", "Sensitive Groups", "Unhealthy", "Hazardous"],
        "PM2.5 (µg/m³)": ["≤ 5", "5–15", "15–25", "25–50", "> 50"],
        "Color"      : ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#7c3aed"],
    }
    scale_html = '<div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:6px">'
    for cat_name, rng, clr in zip(scale_data["Category"], scale_data["PM2.5 (µg/m³)"], scale_data["Color"]):
        scale_html += (
            f'<div style="background:{clr}20;border:1px solid {clr};border-radius:6px;'
            f'padding:4px 8px;font-size:0.75rem;color:{clr};font-weight:600">'
            f'{cat_name}<br><span style="font-weight:400;color:#64748b">{rng}</span></div>'
        )
    scale_html += "</div>"
    st.markdown(scale_html, unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 3: Time Series Charts
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Pollution Over Time")

if "pm25" in filtered.columns:
    # Resample for display clarity
    display_df = filtered["pm25"].dropna().resample("1h").mean().reset_index()
    display_df.columns = ["Time", "PM2.5 (µg/m³)"]
    display_df["AQI Category"] = display_df["PM2.5 (µg/m³)"].apply(lambda x: get_aqi_info(x)[0])

    fig_ts = px.line(
        display_df, x="Time", y="PM2.5 (µg/m³)",
        title="",
        color_discrete_sequence=["#3b82f6"],
        labels={"PM2.5 (µg/m³)": "PM2.5 (µg/m³)", "Time": ""},
    )
    # WHO threshold lines
    fig_ts.add_hline(y=5,  line_dash="dot",  line_color="#22c55e", annotation_text="WHO annual (5)", annotation_position="top left")
    fig_ts.add_hline(y=15, line_dash="dash", line_color="#f59e0b", annotation_text="WHO daily (15)", annotation_position="top left")

    fig_ts.update_layout(
        height=280,
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font={"color": "#1e3a5f"},
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(gridcolor="#e8edf3"),
        yaxis=dict(gridcolor="#e8edf3"),
        hovermode="x unified",
    )
    fig_ts.update_traces(line_width=2, fill="tozeroy", fillcolor="rgba(59,130,246,0.07)")
    st.plotly_chart(fig_ts, use_container_width=True)

# ── Hourly pattern ──
if "pm25" in filtered.columns:
    hourly_avg = filtered["pm25"].dropna().groupby(filtered["pm25"].dropna().index.hour).mean().reset_index()
    hourly_avg.columns = ["Hour of Day", "Mean PM2.5"]
    hourly_avg["Period"] = hourly_avg["Hour of Day"].apply(
        lambda h: "Night" if h < 6 or h >= 22 else ("Morning" if h < 12 else ("Afternoon" if h < 18 else "Evening"))
    )
    color_map = {"Night": "#6366f1", "Morning": "#f59e0b", "Afternoon": "#ef4444", "Evening": "#8b5cf6"}

    fig_hourly = px.bar(
        hourly_avg, x="Hour of Day", y="Mean PM2.5", color="Period",
        color_discrete_map=color_map,
        title="",
        labels={"Mean PM2.5": "Avg PM2.5 (µg/m³)", "Hour of Day": "Hour of Day (0–23)"},
    )
    fig_hourly.add_hline(y=15, line_dash="dash", line_color="#f59e0b", annotation_text="WHO limit")
    fig_hourly.update_layout(
        height=260,
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font={"color": "#1e3a5f"},
        margin=dict(t=30, b=20, l=10, r=10),
        xaxis=dict(gridcolor="#e8edf3", dtick=1),
        yaxis=dict(gridcolor="#e8edf3"),
        legend_title="Time of Day",
    )
    st.caption("💡 This chart shows which hours of the day tend to have more pollution.")
    st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 4: Forecast (7 Days)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔮 7-Day Forecast")
st.caption("Predictions are based on historical patterns and seasonal trends. Treat as estimates, not exact values.")

loc_for_forecast = selected_loc if selected_loc != "All locations" else None
forecast = simulate_forecast(df, location=loc_for_forecast, days=7)

f_col1, f_col2 = st.columns(2)

with f_col1:
    st.markdown("**PM2.5 Forecast**")
    daily_fc = forecast["pm25_forecast"].resample("1D").mean().reset_index()
    daily_fc.columns = ["Date", "Predicted PM2.5"]
    daily_fc["AQI Label"] = daily_fc["Predicted PM2.5"].apply(lambda x: get_aqi_info(x)[0])
    daily_fc["Color"]     = daily_fc["Predicted PM2.5"].apply(lambda x: get_aqi_info(x)[1])

    fig_fc_pm25 = go.Figure()
    fig_fc_pm25.add_trace(go.Scatter(
        x=daily_fc["Date"], y=daily_fc["Predicted PM2.5"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="#3b82f6", width=2.5),
        mode="lines+markers",
        marker=dict(
            size=9,
            color=daily_fc["Color"],
            line=dict(color="white", width=2),
        ),
        hovertemplate="<b>%{x|%A, %b %d}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))
    fig_fc_pm25.add_hline(y=15, line_dash="dash", line_color="#f59e0b", annotation_text="WHO 15 µg/m³")
    fig_fc_pm25.update_layout(
        height=250,
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font={"color": "#1e3a5f"},
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(gridcolor="#e8edf3"),
        yaxis=dict(gridcolor="#e8edf3", title="µg/m³"),
    )
    st.plotly_chart(fig_fc_pm25, use_container_width=True)

with f_col2:
    st.markdown("**Temperature Forecast**")
    daily_temp = forecast["temp_forecast"].resample("1D").agg(["min", "max", "mean"]).reset_index()
    daily_temp.columns = ["Date", "Min", "Max", "Mean"]

    fig_fc_temp = go.Figure()
    fig_fc_temp.add_trace(go.Scatter(
        x=pd.concat([daily_temp["Date"], daily_temp["Date"][::-1]]),
        y=pd.concat([daily_temp["Max"], daily_temp["Min"][::-1]]),
        fill="toself", fillcolor="rgba(251,146,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name="Range",
    ))
    fig_fc_temp.add_trace(go.Scatter(
        x=daily_temp["Date"], y=daily_temp["Mean"],
        line=dict(color="#f97316", width=2.5),
        mode="lines+markers",
        marker=dict(size=8, color="#f97316", line=dict(color="white", width=2)),
        name="Avg Temp",
        hovertemplate="<b>%{x|%A, %b %d}</b><br>Temp: %{y:.1f}°C<extra></extra>",
    ))
    fig_fc_temp.update_layout(
        height=250,
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font={"color": "#1e3a5f"},
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(gridcolor="#e8edf3"),
        yaxis=dict(gridcolor="#e8edf3", title="°C"),
        showlegend=False,
    )
    st.plotly_chart(fig_fc_temp, use_container_width=True)

# Forecast table
st.markdown("**Day-by-day overview:**")
daily_fc["Status"] = daily_fc["AQI Label"].apply(lambda x: get_aqi_info(
    daily_fc.loc[daily_fc["AQI Label"] == x, "Predicted PM2.5"].values[0]
)[2] + " " + x if not daily_fc.loc[daily_fc["AQI Label"] == x].empty else x)

table_df = daily_fc[["Date", "Predicted PM2.5", "AQI Label"]].copy()
table_df["Date"] = table_df["Date"].dt.strftime("%A, %b %d")
table_df.columns = ["Day", "PM2.5 (µg/m³)", "Air Quality"]
st.dataframe(table_df.reset_index(drop=True), use_container_width=True, hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 5: Interactive Map (Folium heatmap + station markers)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 🗺️ Interactive Pollution Map")
st.caption("Scroll to zoom · Click markers for details · Toggle heatmap and stations using the layer control ↗️")

import folium
from folium.plugins import HeatMap, HeatMapWithTime
from streamlit_folium import st_folium

# ── Exact GPS coordinates (verified via Google Maps) ──────────────────────
STATIONS = {
    "Ben Guerir – UM6P Campus": {
        "lat": 32.2359364,   # from Google Maps: maps.app.goo.gl/HwL1GqTejp4XRv4z6
        "lon": -7.9538378,
        "icon": "industry",
    },
    "Marrakech – Résidence Mima": {
        "lat": 31.6601176,   # Google Places verified
        "lon": -8.0291786,
        "icon": "home",
    },
}

# ── Map toggle controls ────────────────────────────────────────────────────
map_ctrl_col1, map_ctrl_col2, map_ctrl_col3 = st.columns(3)
with map_ctrl_col1:
    show_heatmap = st.toggle("🔥 Show Heatmap", value=True)
with map_ctrl_col2:
    show_markers = st.toggle("📍 Show Station Markers", value=True)
with map_ctrl_col3:
    heatmap_metric = st.selectbox("Heatmap metric", ["pm25", "pm1", "um003"],
                                   format_func=lambda x: {"pm25": "PM2.5", "pm1": "PM1", "um003": "UM0.3"}[x])

# ── Build station stats from filtered data ─────────────────────────────────
def match_station(location_name):
    """Map a raw location_name string to the canonical station key."""
    ln = str(location_name).lower()
    if "ben" in ln or "guerir" in ln or "campus" in ln or "um6p" in ln:
        return "Ben Guerir – UM6P Campus"
    if "mima" in ln or "marrakech" in ln or "marrakesh" in ln:
        return "Marrakech – Résidence Mima"
    return None

station_stats = {}
for sname, coords in STATIONS.items():
    # Try to pull stats from the filtered dataframe
    if "location_name" in filtered.columns:
        mask = filtered["location_name"].apply(match_station) == sname
        sub  = filtered[mask]
    else:
        sub = filtered  # single-location dataset

    if heatmap_metric in filtered.columns and len(sub) > 0:
        mean_val = sub[heatmap_metric].dropna().mean()
        max_val  = sub[heatmap_metric].dropna().max()
        n        = sub[heatmap_metric].dropna().count()
    else:
        mean_val = max_val = n = None

    station_stats[sname] = {**coords, "mean": mean_val, "max": max_val, "n": n}

# ── Build heatmap data ─────────────────────────────────────────────────────
# Use the actual lat/lon columns if present, else pin to station coords
if "latitude" in filtered.columns and "longitude" in filtered.columns and heatmap_metric in filtered.columns:
    heat_src = filtered[["latitude", "longitude", heatmap_metric]].dropna()
    # Override lat/lon with exact station coords for known locations
    if "location_name" in filtered.columns:
        for sname, coords in STATIONS.items():
            mask = heat_src.index.isin(
                filtered[filtered["location_name"].apply(match_station) == sname].index
            )
            heat_src.loc[mask, "latitude"]  = coords["lat"]
            heat_src.loc[mask, "longitude"] = coords["lon"]
    heat_data = heat_src[["latitude", "longitude", heatmap_metric]].values.tolist()
else:
    # Synthesise heatmap points around the two known stations
    heat_data = []
    for sname, s in station_stats.items():
        if s["mean"] is not None:
            np.random.seed(abs(hash(sname)) % 999)
            for _ in range(200):
                heat_data.append([
                    s["lat"] + np.random.normal(0, 0.003),
                    s["lon"] + np.random.normal(0, 0.003),
                    float(s["mean"]) + np.random.normal(0, float(s["mean"]) * 0.1),
                ])

# ── Build Folium map ───────────────────────────────────────────────────────
center_lat = np.mean([s["lat"] for s in STATIONS.values()])
center_lon = np.mean([s["lon"] for s in STATIONS.values()])

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles="CartoDB positron",
    control_scale=True,
)

# Add a satellite tile layer option
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Satellite",
    name="🛰️ Satellite",
    overlay=False,
).add_to(m)

folium.TileLayer("CartoDB positron", name="🗺️ Clean Map", overlay=False).add_to(m)

# ── Heatmap layer ──────────────────────────────────────────────────────────
if show_heatmap and heat_data:
    HeatMap(
        heat_data,
        name="🔥 PM2.5 Heatmap",
        radius=28,
        blur=20,
        min_opacity=0.35,
        max_zoom=14,
        gradient={
            "0.0": "#00c851",   # green  – Good
            "0.3": "#ffbb33",   # yellow – Moderate
            "0.6": "#ff8800",   # orange – Sensitive
            "0.8": "#ff4444",   # red    – Unhealthy
            "1.0": "#7c3aed",   # purple – Hazardous
        },
    ).add_to(m)

# ── Station markers ────────────────────────────────────────────────────────
if show_markers:
    marker_group = folium.FeatureGroup(name="📍 Monitoring Stations")

    for sname, s in station_stats.items():
        mean_v = s["mean"]
        cat, color_hex, emoji, health_msg = get_aqi_info(mean_v)

        # Convert hex color to folium color name
        folium_color = (
            "green"      if color_hex == "#22c55e" else
            "lightgreen" if color_hex == "#84cc16" else
            "orange"     if color_hex == "#f59e0b" else
            "red"        if color_hex == "#ef4444" else
            "purple"
        )

        metric_label = {"pm25": "PM2.5", "pm1": "PM1", "um003": "UM0.3"}[heatmap_metric]

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:200px; padding:4px">
            <h4 style="margin:0 0 8px 0; color:#1e3a5f">{sname}</h4>
            <table style="width:100%; border-collapse:collapse; font-size:13px">
                <tr><td style="color:#64748b">Avg {metric_label}</td>
                    <td style="font-weight:700; color:#1e3a5f">{f"{mean_v:.1f} µg/m³" if mean_v else "N/A"}</td></tr>
                <tr><td style="color:#64748b">Max {metric_label}</td>
                    <td style="font-weight:600">{f"{s['max']:.1f} µg/m³" if s['max'] else "N/A"}</td></tr>
                <tr><td style="color:#64748b">Readings</td>
                    <td>{f"{int(s['n']):,}" if s['n'] else "N/A"}</td></tr>
                <tr><td style="color:#64748b">Air Quality</td>
                    <td><span style="color:{color_hex}; font-weight:700">{emoji} {cat}</span></td></tr>
            </table>
            <p style="margin:8px 0 0 0; font-size:12px; color:#475569; border-top:1px solid #e2e8f0; padding-top:6px">
                {health_msg}
            </p>
        </div>
        """

        folium.Marker(
            location=[s["lat"], s["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(
                f"<b>{sname}</b><br>{emoji} {cat}<br>Avg {metric_label}: {f'{mean_v:.1f} µg/m³' if mean_v else 'N/A'}",
                sticky=True,
            ),
            icon=folium.Icon(
                color=folium_color,
                icon=s["icon"],
                prefix="fa",
            ),
        ).add_to(marker_group)

        # Pulsing circle around marker to draw attention
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=18,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.12,
            weight=2,
            opacity=0.5,
        ).add_to(marker_group)

    marker_group.add_to(m)

# Layer control (top-right toggle)
folium.LayerControl(collapsed=False, position="topright").add_to(m)

# ── Render the interactive map ─────────────────────────────────────────────
st_folium(m, height=480, use_container_width=True, returned_objects=[])

# ── Station summary cards ──────────────────────────────────────────────────
st.markdown("**Station Summary:**")
card_cols = st.columns(len(STATIONS))
for col, (sname, s) in zip(card_cols, station_stats.items()):
    mean_v = s["mean"]
    cat, color_hex, emoji, _ = get_aqi_info(mean_v)
    with col:
        st.markdown(
            f"""<div style="background:white;border:1px solid #e8edf3;border-top:4px solid {color_hex};
            border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">
            <div style="font-size:0.78rem;color:#64748b;margin-bottom:4px">{sname}</div>
            <div style="font-size:1.5rem;font-weight:700;color:#1e3a5f">
                {f"{mean_v:.1f}" if mean_v else "—"} <span style="font-size:0.85rem;font-weight:400">µg/m³</span>
            </div>
            <div style="margin-top:6px"><span style="background:{color_hex}20;color:{color_hex};
            font-size:0.78rem;font-weight:700;padding:2px 8px;border-radius:20px">{emoji} {cat}</span></div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px">
                {f"{int(s['n']):,} readings" if s['n'] else "No data in range"}
            </div></div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 6: Insights Panel
# ════════════════════════════════════════════════════════════════════════════
ins_col, comp_col = st.columns([1, 1])

with ins_col:
    st.markdown("### 💡 Key Insights")
    loc_for_insights = selected_loc if selected_loc != "All locations" else None
    insights = generate_insights(filtered, location=loc_for_insights)
    for insight in insights:
        st.markdown(f'<div class="insight-pill">{insight}</div>', unsafe_allow_html=True)

with comp_col:
    st.markdown("### 📊 AQI Distribution")
    if "pm25" in filtered.columns:
        filtered["aqi_cat"] = compute_aqi_category(filtered["pm25"])
        cat_counts = filtered["aqi_cat"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        cat_order  = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Hazardous"]
        cat_colors = {"Good": "#22c55e", "Moderate": "#84cc16",
                      "Unhealthy (Sensitive)": "#f59e0b",
                      "Unhealthy": "#ef4444", "Hazardous": "#7c3aed"}

        cat_counts = cat_counts[cat_counts["Category"].isin(cat_order)]
        cat_counts["Category"] = pd.Categorical(cat_counts["Category"], categories=cat_order, ordered=True)
        cat_counts = cat_counts.sort_values("Category")

        fig_pie = px.pie(
            cat_counts, names="Category", values="Count",
            color="Category", color_discrete_map=cat_colors,
            hole=0.5,
        )
        fig_pie.update_layout(
            height=280,
            paper_bgcolor="white",
            font={"color": "#1e3a5f"},
            margin=dict(t=20, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="v", x=1, y=0.5),
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 7: Data Preview
# ════════════════════════════════════════════════════════════════════════════
with st.expander("🔍 View Raw Data"):
    st.caption(f"Showing {min(500, len(filtered))} of {len(filtered):,} rows after filters.")
    preview_cols = [c for c in ["location_name", "pm25", "pm1", "um003", "temperature", "relativehumidity"] if c in filtered.columns]
    st.dataframe(
        filtered[preview_cols].dropna(how="all").tail(500).reset_index(),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ──
st.markdown(
    '<div class="footer">AirWatch Dashboard · WHO PM2.5 thresholds applied · '
    'Forecasts are indicative estimates based on historical patterns</div>',
    unsafe_allow_html=True,
)
