

> Real-time PM2.5 forecasting for the Marrakech / Ben Guerir area — powered by XGBoost, LSTM, and a live Streamlit dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project collects, cleans, and models air quality data from **AirGradient** sensors and the **OpenAQ API** across two locations in the Marrakech-Safi region of Morocco. It delivers:

- **24-hour PM2.5 forecasts** using XGBoost (R² = 0.988) and LSTM
- **Spatio-temporal heatmaps** of pollution across sensor locations
- **WHO alert banners** when PM2.5 exceeds safe thresholds
- **SHAP explainability charts** to understand model decisions
- A **live Streamlit dashboard** deployable in one click

---

##  Dataset

| Field | Details |
|---|---|
| **Sources** | AirGradient (campus sensors) + OpenAQ API |
| **Locations** | Marrakech-Residence Mima · ARC Air (Ben Guerir) |
| **Period** | March 2025 – April 2026 |
| **Rows** | ~15,000 hourly observations |
| **Features** | PM1, PM2.5, relative humidity, temperature, particle count (um003), lat/lon |

### PM2.5 Distribution (WHO categories)

| Category | Range (µg/m³) | Count |
|---|---|---|
| Good | 0–5 | 2,172 |
| Moderate | 5–15 | 5,542 |
| Unhealthy for Sensitive Groups | 15–25 | 3,712 |
| Unhealthy | 25–50 | 3,440 |
| Very Unhealthy | 50–75 | 227 |
| Hazardous | 75+ | 31 |

---

##  Project Structure

```
air-quality-dashboard/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── preprocessing.py        # Data cleaning & feature engineering pipeline
├── train_model.py          # XGBoost + LSTM training scripts
├── xgboost_model.pkl       # Trained XGBoost model (joblib)
├── requirements.txt        # Python dependencies
├── README.md
│
├── data/
│   └── openaq_merged.csv   # Merged sensor dataset
│
├── notebooks/
│   └── exploration.ipynb   # EDA & model comparison notebook
│
└── utils/
    └── fetch_openaq.py     # OpenAQ API data fetcher
```

---

## Feature Engineering

All features are computed in `preprocessing.py`:

| Feature Group | Features |
|---|---|
| **Raw sensors** | pm1, relativehumidity, temperature, um003 |
| **Rolling averages** | pm25/pm1/rh × {1h, 6h, 24h} windows |
| **Lag features** | pm25 lagged {1h, 3h, 6h, 24h} |
| **Cyclical time** | sin/cos encoding of hour, day-of-week, month |
| **Categorical time** | hour, dayofweek, month, is_weekend |
| **Spatial** | latitude, longitude |

---

## Models

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² |
|---|---|---|---|
| **XGBoost** | **1.53** | **0.37** | **0.988** |
| Random Forest | ~2.1 | ~0.6 | ~0.97 |
| LSTM (24-step) | ~1.8 | ~0.5 | ~0.98 |

### Top Feature Importances 

```
pm1_roll_1h      ████████████  43.3%
pm25_roll_1h     ██████        24.6%
pm1              █████         19.1%
um003            ██             6.8%
rh_roll_24h      ░              0.9%
```
