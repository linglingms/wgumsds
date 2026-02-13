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
