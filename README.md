# WGU Data Science Projects

This repository contains coursework and deployment artifacts from the WGU Master's in Data Science program.

## Active Streamlit Deployment

The repository is currently configured to deploy the D603 Task 3 time series modeling app.

- Streamlit entrypoint: `D603/task3_app.py`
- Dataset: `D603/d603task3_cleaned_data.csv`
- Committed notebook forecast: `D603/d603task3_forecast.csv`
- Documentation assets: `D603/D603 Task 3.docx` and `D603/D603 Task 3 Time Series Modeling.pdf`

## Repository Highlights

- `D600/`: statistical data mining notebooks and datasets
- `D602/`: deployment coursework
- `D603/`: machine learning projects, including the deployed time series app
- `D604/`: separate Streamlit apps for neural network tasks
- `D606/`: capstone work

## Local Setup

Recommended Python version: `3.11`

Install the deployment dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the D603 Streamlit app:

```bash
streamlit run D603/task3_app.py
```

## Streamlit Cloud

Use these settings when creating or updating the Streamlit Cloud app:

- Repository: this repository
- Branch: `main`
- Main file path: `D603/task3_app.py`
- Python version: `3.11`

The root `requirements.txt` and `runtime.txt` are aligned to that deployment target.

