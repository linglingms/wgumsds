# Clustering Dashboard (D603 Task 2)

This repository contains a small Streamlit app to interactively explore clustering on the D603 Task 2 data.

Files added:
- `cluster_dashboard.py` — the Streamlit dashboard app
- `requirements.txt` — Python dependencies

Usage

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the dashboard:

```bash
streamlit run cluster_dashboard.py
```

Notes
- The app attempts to load `d603task2_cleaned_data.csv` first. If not present it will try `medical_clean_d603.xlsx`.
- If your data is already standardized (mean≈0, std≈1) the app will detect it and skip scaling by default.
# Access to Care Dashboard

This repository contains a Streamlit application that visualizes the `Access_to_Care_Dataset.csv` file. The dashboard provides an interactive way to explore the data with filters, charts, and summary statistics.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place the dataset**
   Ensure `Access_to_Care_Dataset.csv` is in the same directory as `app.py`.

3. **Run the app locally**
   ```bash
   streamlit run app.py
   ```

4. **Deploying**
   - You can deploy the app on [Streamlit Cloud](https://streamlit.io/cloud) by linking this repository.
   - Alternatively, use platforms like Heroku, AWS, or Docker with a `Procfile` or `Dockerfile`.

## Features

- Filter data by `TOPIC`, `SUBGROUP`, and `CLASSIFICATION` using sidebar controls.
- Interactive box plots, time-series trends, heatmaps, and summary statistics.

---

*Created automatically by GitHub Copilot.*
