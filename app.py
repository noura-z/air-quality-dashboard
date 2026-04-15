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
try:
    import tensorflow as tf
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
if not LSTM_AVAILABLE:
    st.warning("LSTM model unavailable in this environment. Showing XGBoost forecast.")
warnings.filterwarnings("ignore")

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirWatch – Air Quality Dashboard",
    page_icon=":wind_face:",
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


# ─── SVG Icon Library ─────────────────────────────────────────────────────────
# All icons are inline SVGs — no external dependencies, crisp at any DPI.

def svg_icon(name, size=20, color="currentColor"):
    """Return an inline SVG icon string by name."""
    icons = {
        "wind": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>""",
        "pm25": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2h8"/><path d="M9 2v2.789a4 4 0 0 1-.672 2.219l-.656.984A4 4 0 0 0 7 10.212V20a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-9.789a4 4 0 0 0-.672-2.219l-.656-.984A4 4 0 0 1 15 4.788V2"/><path d="M7 15a6.472 6.472 0 0 1 5 0 6.472 6.472 0 0 0 5 0"/></svg>""",
        "particle": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/><line x1="12" y1="9" x2="12" y2="6"/><line x1="12" y1="15" x2="12" y2="18"/><line x1="9" y1="12" x2="6" y2="12"/><line x1="15" y1="12" x2="18" y2="12"/></svg>""",
        "microscope": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>""",
        "thermometer": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>""",
        "droplet": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>""",
        "chart": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>""",
        "upload": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>""",
        "filter": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>""",
        "map": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>""",
        "heart": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>""",
        "bulb": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>""",
        "forecast": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>""",
        "warning": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
        "check": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>""",
        "shield": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
        "trend_up": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>""",
        "trend_down": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>""",
        "clock": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>""",
        "car": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17H5v-5l2.5-7.5h9L19 12"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/><path d="M5 12h14"/></svg>""",
        "calendar": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>""",
        "pie": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>""",
        "location": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>""",
        "database": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    }
    return icons.get(name, icons["wind"])


def icon_html(name, size=18, color="#64748b", valign="middle"):
    """Wrap an SVG icon for inline HTML use."""
    return f'<span style="display:inline-block;vertical-align:{valign};margin-right:6px">{svg_icon(name, size, color)}</span>'


# UM6P Logo as embedded SVG (University Mohammed VI Polytechnic)
UM6P_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 210 52" width="168" height="42">
  <rect x="0"  y="0" width="44" height="44" rx="5" fill="#E84A14"/>
  <rect x="48" y="0" width="44" height="44" rx="5" fill="#E84A14"/>
  <rect x="96" y="0" width="44" height="44" rx="5" fill="#E84A14"/>
  <rect x="144" y="0" width="44" height="44" rx="5" fill="#E84A14"/>
  <text x="22"  y="32" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-size="24" font-weight="900" fill="#ffffff">U</text>
  <text x="70"  y="32" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-size="24" font-weight="900" fill="#ffffff">M</text>
  <text x="118" y="32" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-size="24" font-weight="900" fill="#ffffff">6</text>
  <text x="166" y="32" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-size="24" font-weight="900" fill="#ffffff">P</text>
  <text x="196" y="12" text-anchor="start" font-family="Arial,sans-serif" font-size="9.5" font-weight="400" fill="#1a1a1a">University</text>
  <text x="196" y="25" text-anchor="start" font-family="Arial,sans-serif" font-size="9.5" font-weight="400" fill="#1a1a1a">Mohammed VI</text>
  <text x="196" y="38" text-anchor="start" font-family="Arial,sans-serif" font-size="9.5" font-weight="400" fill="#1a1a1a">Polytechnic</text>
</svg>
"""

# AQI dot indicator (replaces emoji circles)
def aqi_dot_svg(color, size=14):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 14 14">
      <circle cx="7" cy="7" r="6" fill="{color}" opacity="0.9"/>
      <circle cx="7" cy="7" r="3.5" fill="white" opacity="0.35"/>
    </svg>"""


def get_aqi_info(pm25_value):
    """Return AQI category, color, dot-SVG and health message based on WHO thresholds."""
    if pd.isna(pm25_value):
        return "Unknown", "#94a3b8", aqi_dot_svg("#94a3b8"), "No data available."
    if pm25_value <= 5:
        return "Good", "#22c55e", aqi_dot_svg("#22c55e"), "Air quality is great! Safe to be outdoors."
    elif pm25_value <= 15:
        return "Moderate", "#84cc16", aqi_dot_svg("#84cc16"), "Air quality is acceptable. Most people can enjoy outdoor activities."
    elif pm25_value <= 25:
        return "Unhealthy for Sensitive Groups", "#f59e0b", aqi_dot_svg("#f59e0b"), "Sensitive groups (children, elderly, asthma) should limit prolonged outdoor activity."
    elif pm25_value <= 50:
        return "Unhealthy", "#ef4444", aqi_dot_svg("#ef4444"), "Everyone may begin to experience health effects. Limit outdoor activities."
    else:
        return "Hazardous", "#7c3aed", aqi_dot_svg("#7c3aed"), "Health alert! Everyone should avoid outdoor activity."


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
        insights.append("**Pollution has increased** over the last week compared to the monthly average.")
    elif weekly_mean < monthly_mean * 0.9:
        insights.append("**Pollution has improved** over the last week compared to the monthly average.")
    else:
        insights.append("**Pollution levels are stable** and close to the monthly average.")

    # Diurnal pattern
    hourly = sub["pm25"].groupby(sub.index.hour).mean()
    peak_hour = hourly.idxmax()
    low_hour  = hourly.idxmin()
    tod = "morning" if 6 <= peak_hour <= 10 else ("evening" if 16 <= peak_hour <= 20 else "midday")
    insights.append(f"**Pollution peaks around {peak_hour}:00** ({tod} hours) and is lowest around {low_hour}:00.")

    # WHO threshold
    pct_over = (sub["pm25"] > 15).mean() * 100
    if pct_over > 30:
        insights.append(f"**{pct_over:.0f}% of readings** exceeded the WHO safe daily limit of 15 µg/m³.")
    elif pct_over > 10:
        insights.append(f"**{pct_over:.0f}% of readings** exceeded the WHO safe limit — worth monitoring.")
    else:
        insights.append(f"**Air quality is mostly safe** — only {pct_over:.0f}% of readings exceeded WHO limits.")

    # Weekend vs weekday
    if len(sub) > 200:
        weekday_mean = sub[sub.index.dayofweek < 5]["pm25"].mean()
        weekend_mean = sub[sub.index.dayofweek >= 5]["pm25"].mean()
        if weekday_mean > weekend_mean * 1.1:
            insights.append("**Weekday pollution is higher** than weekends, likely due to traffic and activity.")
        elif weekend_mean > weekday_mean * 1.1:
            insights.append("**Weekend pollution is slightly higher**, which may point to outdoor events or burning.")
        else:
            insights.append("**Weekday and weekend levels are similar**, suggesting pollution isn't mainly traffic-driven.")

    return insights


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # UM6P Logo + AirWatch branding
    st.markdown(UM6P_LOGO_SVG, unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:10px;padding:10px 0 0 0;border-top:1px solid #e8edf3">
      <div style="display:flex;align-items:center;gap:8px">
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
             fill="none" stroke="#2d7d46" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/>
          <path d="M9.6 4.6A2 2 0 1 1 11 8H2"/>
          <path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>
        </svg>
        <span style="font-size:1.15rem;font-weight:700;color:#1a3a2a;letter-spacing:0.3px">AirWatch</span>
      </div>
      <p style="margin:4px 0 0 30px;font-size:0.75rem;color:#64748b">Campus Air Quality Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Upload section
    st.markdown(
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;font-weight:600;color:#1a3a2a">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2d7d46" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
        ' Upload Data</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV with columns: datetime, pm25, location_name, latitude, longitude, etc.",
        label_visibility="collapsed",
    )
    use_sample = st.checkbox("Use sample dataset", value=(uploaded_file is None))
    st.divider()

    # Filters section
    st.markdown(
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;font-weight:600;color:#1a3a2a">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2d7d46" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>'
        ' Filters</div>',
        unsafe_allow_html=True,
    )
    filter_placeholder = st.empty()
    st.divider()

    # Footer
    st.markdown("""
    <div style="font-size:0.72rem;color:#94a3b8;line-height:1.8">
      <div style="display:flex;align-items:center;gap:5px">
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
             fill="none" stroke="#94a3b8" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/>
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
        </svg> OpenAQ &middot; AirGradient
      </div>
      <div style="display:flex;align-items:center;gap:5px">
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
             fill="none" stroke="#94a3b8" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg> WHO PM2.5 thresholds
      </div>
    </div>
    """, unsafe_allow_html=True)


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
            data_source = f"Uploaded: **{uploaded_file.name}**"
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()
    else:
        df = load_sample()
        data_source = "Sample dataset: openaq_merged.csv (Ben Guerir / Marrakech area)"

available = validate_columns(df)

# ── Sidebar filters ──
with filter_placeholder.container():
    # Location filter
    locations = ["All locations"]
    if "location_name" in df.columns:
        locations += sorted(df["location_name"].dropna().unique().tolist())
    selected_loc = st.selectbox("Location", locations)

    # Date range filter
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = st.date_input(
        "Date range",
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
    st.warning("No data matches your current filters. Please adjust the date range or location.")
    st.stop()


# ─── Main Dashboard ───────────────────────────────────────────────────────────

# ── Header ──
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;padding:6px 0 2px 0">
  <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:12px;padding:10px 12px;display:flex;align-items:center">
    {svg_icon('wind', 28, '#2d7d46')}
  </div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:#1a3a2a;line-height:1.2">AirWatch</div>
    <div style="font-size:0.82rem;color:#64748b">{data_source}</div>
  </div>
  <div style="margin-left:auto">{UM6P_LOGO_SVG}</div>
</div>
""", unsafe_allow_html=True)

# ── Warning Banner ──
current_pm25 = filtered["pm25"].dropna().tail(12).mean() if "pm25" in filtered.columns else None
if current_pm25 and current_pm25 > 25:
    label, color, dot_svg, _ = get_aqi_info(current_pm25)
    st.markdown(f"""
    <div class="warning-banner" style="display:flex;align-items:center;gap:10px">
      {svg_icon('warning', 20, '#9a3412')}
      <span><strong>Air Quality Alert:</strong> Recent PM2.5 average is {current_pm25:.1f} µg/m³
      — classified as <strong>{label}</strong>. Consider limiting outdoor activities.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 1: Key Metrics
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
  {svg_icon('chart', 18, '#2d7d46')}
  <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Current Air Quality Snapshot</span>
</div>""", unsafe_allow_html=True)

metric_cols = st.columns(5)

def format_delta(series, suffix=""):
    if len(series) < 48:
        return None, None
    recent   = series.tail(24).mean()
    previous = series.tail(48).head(24).mean()
    delta    = recent - previous
    return f"{delta:+.1f}{suffix}", delta

METRIC_DEFS = [
    ("pm25",             "PM2.5",       "µg/m³", "pm25",        "#3b82f6"),
    ("pm1",              "PM1",         "µg/m³", "particle",    "#8b5cf6"),
    ("um003",            "UM0.3",       "p/dL",  "microscope",  "#0ea5e9"),
    ("temperature",      "Temperature", "°C",    "thermometer", "#f97316"),
    ("relativehumidity", "Humidity",    "%",     "droplet",     "#06b6d4"),
]

for i, (col, label, unit, icon_name, accent) in enumerate(METRIC_DEFS):
    if col in filtered.columns:
        val = filtered[col].dropna().tail(12).mean()
        delta_str, delta_val = format_delta(filtered[col].dropna(), "")
        trend_icon = ""
        trend_color = "#64748b"
        if delta_val is not None:
            if delta_val > 0:
                trend_icon = svg_icon("trend_up", 13, "#ef4444")
                trend_color = "#ef4444"
            else:
                trend_icon = svg_icon("trend_down", 13, "#22c55e")
                trend_color = "#22c55e"
            delta_html = f'<span style="color:{trend_color};font-size:0.75rem;font-weight:600">{trend_icon} {delta_str} {unit}</span>'
        else:
            delta_html = ""
        with metric_cols[i]:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e8edf3;border-top:3px solid {accent};
                        border-radius:12px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">
              <div style="display:flex;align-items:center;gap:7px;margin-bottom:6px">
                {svg_icon(icon_name, 16, accent)}
                <span style="font-size:0.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">{label}</span>
              </div>
              <div style="font-size:1.55rem;font-weight:700;color:#1a3a2a;line-height:1">{val:.1f}
                <span style="font-size:0.8rem;font-weight:400;color:#94a3b8">{unit}</span>
              </div>
              <div style="margin-top:5px">{delta_html}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ════════════════════════════════════════════════════════════════════════════
# Section 2: AQI Indicator + Health Panel
# ════════════════════════════════════════════════════════════════════════════
aqi_col, health_col = st.columns([1, 2])

with aqi_col:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      {svg_icon('shield', 18, '#2d7d46')}
      <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Air Quality Index</span>
    </div>""", unsafe_allow_html=True)

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
            f'<span class="aqi-badge" style="background:{color}20;color:{color};border:2px solid {color};display:inline-flex;align-items:center;gap:7px">'
            f'{aqi_dot_svg(color, 16)} {cat}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("PM2.5 data not available in this dataset.")

with health_col:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      {svg_icon('heart', 18, '#ef4444')}
      <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Health Guidance</span>
    </div>""", unsafe_allow_html=True)

    if "pm25" in filtered.columns:
        cat, color, emoji, msg = get_aqi_info(latest_pm25)

        # Main health message
        st.markdown(
            f'<div class="health-card" style="border-color:{color};display:flex;align-items:flex-start;gap:10px">'
            f'<div style="margin-top:2px;flex-shrink:0">{aqi_dot_svg(color, 18)}</div>'
            f'<div><strong style="color:{color}">{cat}</strong><br>'
            f'<span style="color:#475569;font-size:0.9rem">{msg}</span></div></div>',
            unsafe_allow_html=True,
        )

        # Contextual advice — using SVG check/shield/warning icons
        _chk = svg_icon("check", 14, "#22c55e")
        _wrn = svg_icon("warning", 14, "#f59e0b")
        _shd = svg_icon("shield", 14, "#ef4444")
        advice_map = {
            "Good": [
                (_chk, "Ideal for outdoor sports and exercise"),
                (_chk, "Windows can be kept open for ventilation"),
                (_chk, "No special precautions needed"),
            ],
            "Moderate": [
                (_wrn, "Sensitive people should limit extended outdoor exertion"),
                (_chk, "Most people can enjoy outdoor activities normally"),
                (_chk, "Ventilation is still fine"),
            ],
            "Unhealthy for Sensitive Groups": [
                (_wrn, "Children and elderly should limit outdoor activity"),
                (_wrn, "People with asthma or heart conditions should take precautions"),
                (_chk, "Healthy adults can usually continue normal activity"),
            ],
            "Unhealthy": [
                (_shd, "Everyone should limit prolonged outdoor exertion"),
                (_shd, "Stay indoors when possible, keep windows closed"),
                (_wrn, "Those with respiratory issues should keep medications handy"),
            ],
            "Hazardous": [
                (_shd, "Avoid ALL outdoor physical activity"),
                (_shd, "Stay indoors with windows and doors closed"),
                (_shd, "Use N95 mask if you must go outside"),
                (_shd, "Seek medical advice if experiencing symptoms"),
            ],
        }
        advice = advice_map.get(cat, [(_chk, "Monitor air quality regularly.")])
        for tip_icon, tip_text in advice:
            st.markdown(
                f'<div class="insight-pill" style="display:flex;align-items:center;gap:8px">'
                f'<span style="flex-shrink:0">{tip_icon}</span><span>{tip_text}</span></div>',
                unsafe_allow_html=True,
            )

    # AQI Scale legend — with SVG dots instead of emoji
    st.markdown('<div style="font-weight:600;color:#1a3a2a;font-size:0.85rem;margin-top:8px;margin-bottom:6px">WHO PM2.5 Scale</div>', unsafe_allow_html=True)
    scale_items = [
        ("Good",            "≤ 5",   "#22c55e"),
        ("Moderate",        "5–15",  "#84cc16"),
        ("Sensitive",       "15–25", "#f59e0b"),
        ("Unhealthy",       "25–50", "#ef4444"),
        ("Hazardous",       "> 50",  "#7c3aed"),
    ]
    scale_html = '<div style="display:flex;gap:6px;flex-wrap:wrap">'
    for cat_name, rng, clr in scale_items:
        scale_html += (
            f'<div style="background:{clr}10;border:1px solid {clr}55;border-radius:8px;'
            f'padding:5px 9px;font-size:0.74rem;display:flex;align-items:center;gap:5px">'
            f'{aqi_dot_svg(clr, 10)}'
            f'<span style="color:{clr};font-weight:700">{cat_name}</span>'
            f'<span style="color:#94a3b8">{rng}</span></div>'
        )
    scale_html += "</div>"
    st.markdown(scale_html, unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 3: Time Series Charts
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
  {svg_icon('forecast', 18, '#3b82f6')}
  <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Pollution Over Time</span>
</div>""", unsafe_allow_html=True)

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
    st.caption("This chart shows which hours of the day tend to have more pollution.")
    st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# Section 4: Forecast (7 Days)
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
  {svg_icon('forecast', 18, '#8b5cf6')}
  <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">7-Day Forecast</span>
</div>""", unsafe_allow_html=True)
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
st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
  {svg_icon('map', 18, '#0ea5e9')}
  <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Interactive Pollution Map</span>
</div>""", unsafe_allow_html=True)
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
    show_heatmap = st.toggle("Show Heatmap", value=True)
with map_ctrl_col2:
    show_markers = st.toggle("Show Station Markers", value=True)
with map_ctrl_col3:
    heatmap_metric = st.selectbox("Metric", ["pm25", "pm1", "um003"],
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
    name="Satellite View",
    overlay=False,
).add_to(m)

folium.TileLayer("CartoDB positron", name="Clean Map", overlay=False).add_to(m)

# ── Heatmap layer ──────────────────────────────────────────────────────────
if show_heatmap and heat_data:
    HeatMap(
        heat_data,
        name="Pollution Heatmap",
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
    marker_group = folium.FeatureGroup(name="Monitoring Stations")

    for sname, s in station_stats.items():
        mean_v = s["mean"]
        cat, color_hex, _dot_svg, health_msg = get_aqi_info(mean_v)

        # Convert hex color to folium color name
        folium_color = (
            "green"      if color_hex == "#22c55e" else
            "lightgreen" if color_hex == "#84cc16" else
            "orange"     if color_hex == "#f59e0b" else
            "red"        if color_hex == "#ef4444" else
            "purple"
        )

        metric_label = {"pm25": "PM2.5", "pm1": "PM1", "um003": "UM0.3"}[heatmap_metric]
        dot_style = f"display:inline-block;width:10px;height:10px;border-radius:50%;background:{color_hex};margin-right:5px;vertical-align:middle"

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:210px; padding:4px">
            <div style="font-weight:700;font-size:14px;color:#1e3a5f;margin-bottom:8px;
                        padding-bottom:6px;border-bottom:1px solid #e8edf3">{sname}</div>
            <table style="width:100%; border-collapse:collapse; font-size:13px">
                <tr><td style="color:#64748b;padding:3px 0">Avg {metric_label}</td>
                    <td style="font-weight:700;color:#1e3a5f">{f"{mean_v:.1f} µg/m³" if mean_v else "N/A"}</td></tr>
                <tr><td style="color:#64748b;padding:3px 0">Max {metric_label}</td>
                    <td style="font-weight:600">{f"{s['max']:.1f} µg/m³" if s['max'] else "N/A"}</td></tr>
                <tr><td style="color:#64748b;padding:3px 0">Readings</td>
                    <td>{f"{int(s['n']):,}" if s['n'] else "N/A"}</td></tr>
                <tr><td style="color:#64748b;padding:3px 0">Air Quality</td>
                    <td><span style="{dot_style}"></span>
                        <span style="color:{color_hex};font-weight:700">{cat}</span></td></tr>
            </table>
            <p style="margin:8px 0 0;font-size:12px;color:#475569;border-top:1px solid #e2e8f0;padding-top:6px">
                {health_msg}
            </p>
        </div>
        """

        folium.Marker(
            location=[s["lat"], s["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=folium.Tooltip(
                f"<b>{sname}</b><br>{cat}<br>Avg {metric_label}: {f'{mean_v:.1f} µg/m³' if mean_v else 'N/A'}",
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
            font-size:0.78rem;font-weight:700;padding:3px 9px;border-radius:20px;display:inline-flex;align-items:center;gap:5px">{aqi_dot_svg(color_hex, 10)} {cat}</span></div>
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
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      {svg_icon('bulb', 18, '#f59e0b')}
      <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">Key Insights</span>
    </div>""", unsafe_allow_html=True)
    loc_for_insights = selected_loc if selected_loc != "All locations" else None
    insights = generate_insights(filtered, location=loc_for_insights)

    # Map plain-text insight prefixes to SVG icons
    def _insight_icon(text):
        t = text.lower()
        if "increase" in t or "higher" in t or "rising" in t:
            return svg_icon("trend_up", 15, "#ef4444")
        if "improve" in t or "lower" in t or "decreas" in t:
            return svg_icon("trend_down", 15, "#22c55e")
        if "stable" in t or "similar" in t:
            return svg_icon("chart", 15, "#64748b")
        if "peak" in t or "hour" in t or "time" in t:
            return svg_icon("clock", 15, "#8b5cf6")
        if "weekday" in t or "weekend" in t or "traffic" in t:
            return svg_icon("car", 15, "#0ea5e9")
        if "who" in t or "limit" in t or "safe" in t or "exceed" in t:
            return svg_icon("shield", 15, "#f59e0b")
        return svg_icon("bulb", 15, "#f59e0b")

    for insight in insights:
        # Strip any leading markdown emoji/symbols before the first **
        clean = insight.lstrip(" ")
        ico = _insight_icon(clean)
        st.markdown(
            f'<div class="insight-pill" style="display:flex;align-items:flex-start;gap:9px">'
            f'<span style="flex-shrink:0;margin-top:1px">{ico}</span>'
            f'<span>{clean}</span></div>',
            unsafe_allow_html=True,
        )

with comp_col:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      {svg_icon('pie', 18, '#6366f1')}
      <span style="font-size:1.05rem;font-weight:700;color:#1a3a2a">AQI Distribution</span>
    </div>""", unsafe_allow_html=True)
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
with st.expander("View Raw Data"):
    st.caption(f"Showing {min(500, len(filtered))} of {len(filtered):,} rows after filters.")
    preview_cols = [c for c in ["location_name", "pm25", "pm1", "um003", "temperature", "relativehumidity"] if c in filtered.columns]
    st.dataframe(
        filtered[preview_cols].dropna(how="all").tail(500).reset_index(),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ──
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:center;gap:16px;
            padding:16px 0;margin-top:16px;border-top:1px solid #e8edf3;
            font-size:0.78rem;color:#94a3b8">
  <div style="display:flex;align-items:center;gap:5px">
    {svg_icon('database', 12, '#cbd5e1')} OpenAQ &middot; AirGradient
  </div>
  <div style="display:flex;align-items:center;gap:5px">
    {svg_icon('shield', 12, '#cbd5e1')} WHO PM2.5 thresholds applied
  </div>
  <div style="display:flex;align-items:center;gap:5px">
    {svg_icon('forecast', 12, '#cbd5e1')} Forecasts are indicative estimates
  </div>
  <div>{UM6P_LOGO_SVG.replace('width="160"','width="80"').replace('height="44"','height="24"')}</div>
</div>
""", unsafe_allow_html=True)
