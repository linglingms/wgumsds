# D603 Task 3 Streamlit App

This folder now includes a Streamlit app for the D603 Task 3 time series modeling project.

## App entrypoint

- `D603/task3_app.py`

## What the app shows

- Daily revenue overview from `d603task3_cleaned_data.csv`
- ADF stationarity results
- ACF and PACF diagnostics
- Seasonal decomposition view
- ARIMA holdout evaluation against the final test window
- Future forecast with 95% confidence intervals
- Extracted task documentation and PDF download

## Local run

From the workspace root:

```bash
pip install -r requirements.txt
streamlit run D603/task3_app.py
```

## Deployment target

The repository root deployment files are configured so Streamlit Cloud can launch:

```bash
streamlit run D603/task3_app.py
```