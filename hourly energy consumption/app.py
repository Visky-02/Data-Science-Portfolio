"""
P-702 – Hourly Energy Consumption Forecast
Streamlit deployment application.

The selected model from the five-model comparison is:
Previous-Day Seasonal Naive (24h)

This app:
1. Loads PJM hourly data from an uploaded CSV/Excel file or a local default file.
2. Cleans duplicate timestamps and fills missing hourly observations.
3. Uses the selected 24-hour seasonal-naive forecasting method.
4. Generates a 30-day / 720-hour forecast.
5. Displays summary metrics, forecast table and charts.
6. Allows CSV download.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PJM Hourly Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ PJM Hourly Energy Consumption Forecast")
st.markdown(
    """
    **P-702 Forecasting Project**

    This application provides a 30-day hourly electricity-consumption forecast
    using the **Previous-Day Seasonal Naive (24h)** approach selected from the
    five-model comparison.
    """
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data_from_path(path):
    """Load the PJM dataset from CSV or Excel."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_excel(path, sheet_name="PJMW_hourly")
        except Exception:
            df = pd.read_excel(path)

    required = {"Datetime", "PJMW_MW"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required}. "
            f"Available columns: {list(df.columns)}"
        )

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
    df = df[["Datetime", "PJMW_MW"]].dropna(subset=["Datetime"])
    return df


def clean_hourly_series(df):
    """
    Apply the same core time-series preparation used in the notebook:
    - sort chronologically
    - average duplicate timestamps
    - regularize to hourly frequency
    - time-interpolate missing observations
    """
    df = df.sort_values("Datetime").copy()

    duplicate_count = int(df["Datetime"].duplicated().sum())

    ts = (
        df.groupby("Datetime")["PJMW_MW"]
          .mean()
          .sort_index()
          .asfreq("h")
    )

    missing_count = int(ts.isna().sum())

    ts = ts.interpolate(method="time").ffill().bfill()

    return ts, duplicate_count, missing_count


def forecast_previous_day(ts, forecast_hours):
    """
    Previous-Day Seasonal Naive:
    each future hour uses the value from 24 hours earlier.
    Predictions are appended recursively for the next days.
    """
    future_index = pd.date_range(
        ts.index.max() + pd.Timedelta(hours=1),
        periods=forecast_hours,
        freq="h"
    )

    history = list(ts.astype(float).values)
    predictions = []

    for _ in future_index:
        pred = float(history[-24])
        predictions.append(pred)
        history.append(pred)

    forecast = pd.DataFrame({
        "Datetime": future_index,
        "Forecast_MW": predictions
    })

    return forecast


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Forecast Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload PJM dataset",
    type=["csv", "xlsx"]
)

forecast_days = st.sidebar.number_input(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=30,
    step=1
)

st.sidebar.info(
    "Selected model: Previous-Day Seasonal Naive (24h)"
)

# -----------------------------
# Load data
# -----------------------------
try:
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file, sheet_name="PJMW_hourly")
            except Exception:
                df = pd.read_excel(uploaded_file)

        required = {"Datetime", "PJMW_MW"}
        if not required.issubset(df.columns):
            st.error(
                f"The uploaded file must contain {required}. "
                f"Available columns: {list(df.columns)}"
            )
            st.stop()

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
        df = df[["Datetime", "PJMW_MW"]].dropna(subset=["Datetime"])

    else:
        # Look for a local PJM file in the app folder and common subfolders.
        candidates = [
            "PJMW_hourly.csv",
            "PJMW_MW_Hourly.xlsx",
            "PJMW_hourly.xlsx"
        ]

        local_path = None

        for name in candidates:
            if os.path.exists(name):
                local_path = name
                break

        if local_path is None:
            for pattern in [
                "**/PJMW_hourly.csv",
                "**/PJMW_MW_Hourly.xlsx",
                "**/PJMW_hourly.xlsx"
            ]:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    local_path = matches[0]
                    break

        if local_path is None:
            st.warning(
                "No local PJM dataset was found. Please upload "
                "PJMW_hourly.csv or PJMW_MW_Hourly.xlsx using the sidebar."
            )
            st.stop()

        df = load_data_from_path(local_path)

    # Clean and regularize the time series.
    ts, duplicate_count, missing_count = clean_hourly_series(df)

except Exception as exc:
    st.error(f"Unable to load/process the dataset: {exc}")
    st.stop()

# -----------------------------
# Dataset summary
# -----------------------------
st.subheader("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Observations", f"{len(ts):,}")

with c2:
    st.metric("Start", ts.index.min().strftime("%Y-%m-%d"))

with c3:
    st.metric("End", ts.index.max().strftime("%Y-%m-%d"))

with c4:
    st.metric("Average Load", f"{ts.mean():,.2f} MW")

st.caption(
    f"Duplicate timestamps handled: {duplicate_count} | "
    f"Missing hourly observations filled: {missing_count}"
)

# -----------------------------
# Model selection
# -----------------------------
st.subheader("2. Selected Forecasting Model")

st.success(
    "Previous-Day Seasonal Naive (24h) — selected because it achieved "
    "the lowest test-set MAE, RMSE and MAPE in the five-model comparison."
)

comparison_data = pd.DataFrame({
    "Model": [
        "Previous-Day Seasonal Naive",
        "Previous-Week Seasonal Naive",
        "XGBoost",
        "HistGradientBoosting",
        "Random Forest"
    ],
    "MAE": [382.672, 612.047, 624.689, 661.409, 709.236],
    "RMSE": [501.370, 817.535, 855.264, 894.684, 948.750],
    "MAPE (%)": [6.663, 10.505, 10.634, 10.922, 11.451]
})

comparison_display = comparison_data.copy()
comparison_display["MAE"] = comparison_display["MAE"].round(3)
comparison_display["RMSE"] = comparison_display["RMSE"].round(3)
comparison_display["MAPE (%)"] = comparison_display["MAPE (%)"].round(3)

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Historical data visualization
# -----------------------------
st.subheader("3. Historical Consumption")

history_window = min(len(ts), 24 * 14)

st.line_chart(
    ts.tail(history_window).rename("Historical MW")
)

# -----------------------------
# Generate forecast
# -----------------------------
forecast_hours = int(forecast_days) * 24

if st.button("Generate Forecast", type="primary"):
    forecast = forecast_previous_day(ts, forecast_hours)

    st.subheader("4. Forecast Summary")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Average Forecast",
            f"{forecast['Forecast_MW'].mean():,.2f} MW"
        )

    with m2:
        st.metric(
            "Peak Forecast",
            f"{forecast['Forecast_MW'].max():,.2f} MW"
        )

    with m3:
        st.metric(
            "Minimum Forecast",
            f"{forecast['Forecast_MW'].min():,.2f} MW"
        )

    st.subheader("5. Hourly Forecast")

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("6. Forecast Visualization")

    chart_df = forecast.set_index("Datetime")
    st.line_chart(chart_df["Forecast_MW"])

    # Daily summary
    daily = forecast.copy()
    daily["Forecast_Day"] = (
        np.arange(len(daily)) // 24
    ) + 1

    daily_summary = (
        daily.groupby("Forecast_Day")
             .agg(
                 Start_Datetime=("Datetime", "min"),
                 End_Datetime=("Datetime", "max"),
                 Average_Forecast_MW=("Forecast_MW", "mean"),
                 Minimum_Forecast_MW=("Forecast_MW", "min"),
                 Maximum_Forecast_MW=("Forecast_MW", "max")
             )
             .reset_index()
    )

    for col in [
        "Average_Forecast_MW",
        "Minimum_Forecast_MW",
        "Maximum_Forecast_MW"
    ]:
        daily_summary[col] = daily_summary[col].round(2)

    st.subheader("7. Daily Forecast Summary")
    st.dataframe(
        daily_summary,
        use_container_width=True,
        hide_index=True
    )

    # Download buttons
    hourly_csv = forecast.to_csv(index=False).encode("utf-8")
    daily_csv = daily_summary.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download Hourly Forecast CSV",
            data=hourly_csv,
            file_name="PJMW_Final_30_Day_Forecast.csv",
            mime="text/csv"
        )

    with d2:
        st.download_button(
            "Download Daily Summary CSV",
            data=daily_csv,
            file_name="PJMW_Final_30_Day_Daily_Summary.csv",
            mime="text/csv"
        )

    st.success("Forecast generated successfully.")

else:
    st.info(
        "Provide the dataset using the sidebar if required, choose the "
        "forecast horizon, and click **Generate Forecast**."
    )

st.markdown("---")
st.caption(
    "P-702 Hourly Energy Consumption Forecast | "
    "Five-model comparison with Previous-Day Seasonal Naive selected for deployment"
)
