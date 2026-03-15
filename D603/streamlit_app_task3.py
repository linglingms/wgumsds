from __future__ import annotations

from pathlib import Path
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "d603task3_cleaned_data.csv"
FORECAST_PATH = BASE_DIR / "d603task3_forecast.csv"
DOC_PATH = BASE_DIR / "D603 Task 3.docx"
PDF_PATH = BASE_DIR / "D603 Task 3 Time Series Modeling.pdf"

JARGON_GLOSSARY = {
    "Time series": "A sequence of values measured over time, like daily revenue.",
    "Stationary": "A stable pattern over time where average level and variation stay fairly consistent.",
    "Differencing": "Subtracting each value from the prior value to focus on change rather than raw level.",
    "ARIMA": "A forecasting model that uses recent history and error patterns to predict future values.",
    "ADF test": "A statistical test used to check whether a series is stable enough for time-series modeling.",
    "ACF": "Shows how strongly values relate to earlier values at different time gaps.",
    "PACF": "Shows direct relationships with earlier values after removing indirect effects.",
    "Holdout set": "A final block of data not used for training, reserved for honest model evaluation.",
    "MAE": "Mean absolute error; average absolute prediction miss size.",
    "RMSE": "Root mean squared error; like MAE but penalizes larger misses more heavily.",
    "Confidence interval": "A plausible range for future values, not a guarantee.",
    "Seasonality": "A repeating pattern that occurs on a regular cycle, such as yearly behavior.",
}

TOPIC_KEYWORDS = {
    "overview": ["dataset", "data", "overview", "preparation", "clean", "revenue", "day"],
    "diagnostics": ["stationary", "adf", "acf", "pacf", "diagnostic", "differencing"],
    "forecasts": ["forecast", "arima", "holdout", "rmse", "mae", "confidence", "prediction"],
    "documentation": ["part", "task", "model", "evaluation"],
}


st.set_page_config(
    page_title="D603 Task 3 - Time Series Modeling",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(232, 246, 252, 0.95), transparent 35%),
                linear-gradient(180deg, #f8fbfd 0%, #eef4f7 100%);
        }
        .stApp, .stApp * {
            color: #000000 !important;
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(224, 240, 247, 0.98), rgba(209, 232, 242, 0.94));
            color: #000000;
            box-shadow: 0 16px 40px rgba(19, 65, 82, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.5rem 0 0;
            max-width: 56rem;
            font-size: 1rem;
            line-height: 1.55;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(13, 86, 110, 0.08);
            padding: 0.75rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(18, 57, 71, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def extract_docx_text(docx_path: str) -> str:
    path = Path(docx_path)
    if not path.exists():
        return "Documentation file not found."

    with zipfile.ZipFile(path, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")

    text = re.sub(r"<w:tab\s*/>", "\t", xml)
    text = re.sub(r"</w:p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#x27;", "'")
    )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def select_doc_snippets(doc_text: str, topic: str, max_items: int = 4) -> list[str]:
    if not doc_text or doc_text == "Documentation file not found.":
        return []

    keywords = TOPIC_KEYWORDS.get(topic, [])
    source_lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    matches: list[str] = []

    for line in source_lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in keywords):
            matches.append(line)
        if len(matches) >= max_items:
            break

    return matches


def render_non_technical_guide(terms: list[str], title: str = "Jargon Buster") -> None:
    st.markdown(f"#### {title}")
    guide_rows = {term: JARGON_GLOSSARY[term] for term in terms if term in JARGON_GLOSSARY}
    if guide_rows:
        st.table(pd.DataFrame(guide_rows.items(), columns=["Term", "Plain-English meaning"]))


def render_doc_context(doc_text: str, topic: str, heading: str) -> None:
    st.markdown(f"#### {heading}")
    snippets = select_doc_snippets(doc_text, topic=topic)

    if not snippets:
        st.info("No matching excerpts were found in the documentation for this section.")
        return

    for snippet in snippets:
        st.markdown(f"- {snippet}")


@st.cache_data
def load_time_series(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df = df.dropna(subset=["Day", "Revenue"]).sort_values("Day").reset_index(drop=True)
    df["Revenue"] = df["Revenue"].ffill()
    return df


@st.cache_data
def load_committed_forecast(forecast_path: str) -> pd.DataFrame:
    if not Path(forecast_path).exists():
        return pd.DataFrame(columns=["predicted_mean"])
    return pd.read_csv(forecast_path)


@st.cache_data
def run_analysis(
    data_path: str,
    order: tuple[int, int, int],
    test_size: int,
    future_steps: int,
) -> dict[str, object]:
    df = load_time_series(data_path)
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("Test size must be greater than 0 and smaller than the dataset length.")

    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()

    adf_stat, adf_pvalue, *_ = adfuller(df["Revenue"])
    differenced = df["Revenue"].diff().dropna()

    holdout_fit = ARIMA(train["Revenue"], order=order).fit()
    holdout_result = holdout_fit.get_forecast(steps=len(test))
    holdout_mean = pd.Series(holdout_result.predicted_mean, index=test.index, name="forecast")
    holdout_ci = holdout_result.conf_int(alpha=0.05)
    holdout_ci.index = test.index

    mae = mean_absolute_error(test["Revenue"], holdout_mean)
    rmse = float(np.sqrt(mean_squared_error(test["Revenue"], holdout_mean)))

    full_fit = ARIMA(df["Revenue"], order=order).fit()
    future_result = full_fit.get_forecast(steps=future_steps)
    future_mean = pd.Series(future_result.predicted_mean, name="forecast")
    future_ci = future_result.conf_int(alpha=0.05).reset_index(drop=True)
    last_day = int(df["Day"].iloc[-1])
    future_days = pd.Series(range(last_day + 1, last_day + future_steps + 1), name="Day")

    return {
        "data": df,
        "train": train,
        "test": test,
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "differenced": differenced,
        "holdout_forecast": holdout_mean,
        "holdout_ci": holdout_ci,
        "mae": float(mae),
        "rmse": rmse,
        "future_days": future_days,
        "future_forecast": future_mean.reset_index(drop=True),
        "future_ci": future_ci,
    }


def build_revenue_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["Day"], df["Revenue"], color="#0b4e6c", linewidth=2)
    ax.set_title("Daily Revenue Across the Full Observation Window")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_diagnostics_figure(series: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(series, lags=40, ax=axes[0])
    axes[0].set_title("Autocorrelation Function")
    plot_pacf(series, lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation Function")
    fig.tight_layout()
    return fig


def build_decomposition_figure(df: pd.DataFrame) -> plt.Figure | None:
    try:
        decomposition = seasonal_decompose(df["Revenue"], period=365, extrapolate_trend="freq")
    except ValueError:
        return None

    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    return fig


def build_holdout_figure(analysis: dict[str, object]) -> plt.Figure:
    train = analysis["train"]
    test = analysis["test"]
    holdout_forecast = analysis["holdout_forecast"]
    holdout_ci = analysis["holdout_ci"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["Day"], train["Revenue"], label="Training data", color="#0b4e6c", linewidth=2)
    ax.plot(test["Day"], test["Revenue"], label="Test data", color="#1f7a8c", linewidth=2)
    ax.plot(test["Day"], holdout_forecast.values, label="ARIMA forecast", color="#c65d00", linewidth=2)
    ax.fill_between(
        test["Day"],
        holdout_ci.iloc[:, 0],
        holdout_ci.iloc[:, 1],
        color="#f2c185",
        alpha=0.35,
        label="95% confidence interval",
    )
    ax.set_title("30-Day Holdout Forecast vs Actual Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def build_future_figure(analysis: dict[str, object]) -> plt.Figure:
    df = analysis["data"]
    future_days = analysis["future_days"]
    future_forecast = analysis["future_forecast"]
    future_ci = analysis["future_ci"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Day"], df["Revenue"], label="Observed revenue", color="#0b4e6c", linewidth=2)
    ax.plot(future_days, future_forecast, label="Future forecast", color="#c65d00", linewidth=2)
    ax.fill_between(
        future_days,
        future_ci.iloc[:, 0],
        future_ci.iloc[:, 1],
        color="#f2c185",
        alpha=0.35,
        label="95% confidence interval",
    )
    ax.set_title("Future Revenue Forecast")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def render_documentation(doc_path: Path, pdf_path: Path) -> None:
    st.subheader("Task Documentation")
    documentation_text = extract_docx_text(str(doc_path))

    if documentation_text == "Documentation file not found.":
        st.warning(documentation_text)
        return

    if pdf_path.exists():
        st.download_button(
            label="Download submitted PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
        )

    with st.expander("View extracted documentation text", expanded=False):
        st.text_area("Documentation", value=documentation_text, height=420)


def main() -> None:
    apply_custom_style()
    documentation_text = extract_docx_text(str(DOC_PATH))

    st.markdown(
        """
        <section class="hero">
            <h1>D603 Task 3 Time Series Modeling</h1>
            <p>
                Interactive review of the daily revenue forecasting workflow using the committed Task 3 dataset,
                an ARIMA model, and the same 30-day holdout framing used in the notebook submission.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Controls")
        p_value = st.number_input("ARIMA p", min_value=0, max_value=5, value=2, step=1)
        d_value = st.number_input("ARIMA d", min_value=0, max_value=2, value=1, step=1)
        q_value = st.number_input("ARIMA q", min_value=0, max_value=5, value=1, step=1)
        test_size = st.slider("Holdout length", min_value=14, max_value=60, value=30, step=1)
        future_steps = st.slider("Future forecast horizon", min_value=14, max_value=90, value=30, step=1)

        st.caption(
            "The notebook baseline uses ARIMA(2, 1, 1) with the final 30 days reserved as the holdout set."
        )

        with st.expander("Quick glossary for non-technical readers", expanded=False):
            render_non_technical_guide(
                [
                    "Time series",
                    "ARIMA",
                    "Holdout set",
                    "MAE",
                    "RMSE",
                    "Confidence interval",
                ],
                title="",
            )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    try:
        analysis = run_analysis(
            str(DATA_PATH),
            order=(int(p_value), int(d_value), int(q_value)),
            test_size=int(test_size),
            future_steps=int(future_steps),
        )
    except Exception as exc:
        st.error(f"Could not run the ARIMA analysis: {exc}")
        return

    committed_forecast = load_committed_forecast(str(FORECAST_PATH))
    df = analysis["data"]
    differenced = analysis["differenced"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Observations", f"{len(df):,}")
    metric_cols[1].metric("ADF statistic", f"{analysis['adf_stat']:.4f}")
    metric_cols[2].metric("ADF p-value", f"{analysis['adf_pvalue']:.4f}")
    metric_cols[3].metric("Holdout RMSE", f"{analysis['rmse']:.4f}")

    overview_tab, diagnostics_tab, forecast_tab, docs_tab = st.tabs(
        ["Overview", "Diagnostics", "Forecasts", "Documentation"]
    )

    with overview_tab:
        render_doc_context(documentation_text, topic="overview", heading="From Your Submitted Documentation")
        render_non_technical_guide(
            ["Time series", "Stationary", "Differencing"],
            title="Overview Jargon Buster",
        )

        summary_col, sample_col = st.columns([1.1, 0.9])
        with summary_col:
            st.subheader("Dataset Snapshot")
            st.write(df.head(10))
            st.write(
                {
                    "Minimum Revenue": round(float(df["Revenue"].min()), 4),
                    "Maximum Revenue": round(float(df["Revenue"].max()), 4),
                    "Average Revenue": round(float(df["Revenue"].mean()), 4),
                    "Std Dev": round(float(df["Revenue"].std()), 4),
                }
            )
        with sample_col:
            st.subheader("Committed Forecast Asset")
            if committed_forecast.empty:
                st.info("No committed forecast CSV was found.")
            else:
                st.dataframe(committed_forecast.head(10), use_container_width=True)
                st.caption("This is the forecast CSV committed from the notebook workflow.")

        st.pyplot(build_revenue_figure(df))

        st.download_button(
            label="Download cleaned dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=DATA_PATH.name,
            mime="text/csv",
        )

    with diagnostics_tab:
        render_doc_context(documentation_text, topic="diagnostics", heading="Diagnostic Notes from the Report")
        render_non_technical_guide(
            ["ADF test", "ACF", "PACF", "Seasonality", "Stationary", "Differencing"],
            title="Diagnostics Jargon Buster",
        )

        stationarity_label = "Non-stationary" if analysis["adf_pvalue"] > 0.05 else "Stationary"
        st.write(
            {
                "Stationarity assessment": stationarity_label,
                "Differenced series used for ACF/PACF": bool(analysis["adf_pvalue"] > 0.05),
            }
        )
        if analysis["adf_pvalue"] > 0.05:
            st.info(
                "Interpretation: the baseline pattern drifts over time, so the app compares day-to-day changes"
                " for diagnostics."
            )
        else:
            st.info("Interpretation: the baseline level is stable enough for direct diagnostic plots.")

        series_for_diagnostics = differenced if analysis["adf_pvalue"] > 0.05 else df["Revenue"]
        st.pyplot(build_diagnostics_figure(series_for_diagnostics))

        decomposition_figure = build_decomposition_figure(df)
        if decomposition_figure is not None:
            st.pyplot(decomposition_figure)
        else:
            st.info("Seasonal decomposition was not available for the current series length and period.")

    with forecast_tab:
        render_doc_context(documentation_text, topic="forecasts", heading="Forecasting Highlights from the Report")
        render_non_technical_guide(
            ["ARIMA", "Holdout set", "MAE", "RMSE", "Confidence interval"],
            title="Forecast Jargon Buster",
        )

        forecast_metric_cols = st.columns(3)
        forecast_metric_cols[0].metric("Holdout MAE", f"{analysis['mae']:.4f}")
        forecast_metric_cols[1].metric("Forecast horizon", f"{future_steps} days")
        forecast_metric_cols[2].metric(
            "Final forecasted value",
            f"{float(analysis['future_forecast'].iloc[-1]):.4f}",
        )

        st.pyplot(build_holdout_figure(analysis))
        st.pyplot(build_future_figure(analysis))
        st.caption(
            "Plain-language readout: narrower confidence bands suggest less uncertainty; wider bands suggest"
            " more uncertainty in longer-range predictions."
        )

        future_export = pd.DataFrame(
            {
                "Day": analysis["future_days"],
                "Forecast": analysis["future_forecast"],
                "Lower_95": analysis["future_ci"].iloc[:, 0],
                "Upper_95": analysis["future_ci"].iloc[:, 1],
            }
        )
        st.dataframe(future_export, use_container_width=True)
        st.download_button(
            label="Download future forecast",
            data=future_export.to_csv(index=False).encode("utf-8"),
            file_name="d603task3_future_forecast.csv",
            mime="text/csv",
        )

    with docs_tab:
        render_doc_context(documentation_text, topic="documentation", heading="Auto-Selected Report Excerpts")
        render_non_technical_guide(list(JARGON_GLOSSARY.keys())[:8], title="Core Concepts")
        render_documentation(DOC_PATH, PDF_PATH)


if __name__ == "__main__":
    main()