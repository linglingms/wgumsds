# Clustering Dashboard (D603 Task 2)

Interactive Streamlit dashboard for exploring K-Means and Agglomerative clustering techniques on medical data.

## Files

- `cluster_dashboard.py` — the Streamlit dashboard app
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Local Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run cluster_dashboard.py
```

The app will open at `http://localhost:8501`.

## Features

- Load data from CSV or Excel
- Multi-feature selection for clustering
- Choose between KMeans and Agglomerative clustering
- Adjust number of clusters with slider
- View elbow curve and silhouette scores
- Interactive PCA visualization (2D scatter plot)
- Download labeled data as CSV

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repository:**
   - Go to https://github.com/new
   - Repository name: `clustering-dashboard`
   - Public repo
   - Create

2. **Push code to GitHub:**
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/clustering-dashboard.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Click "Create app"
   - Select your GitHub repo, branch (`main`), and file (`cluster_dashboard.py`)
   - Deploy

**Note:** Add the data file (`d603task2_cleaned_data.csv`) to the repo or configure it to load from a public URL.

## Notes

- The app detects if data is already scaled (mean≈0, std≈1) and skips scaling by default
- Cluster centers are displayed in original data units (when applicable)
- Supports various numeric column selections for flexible analysis
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
