# D604 Task 1 - Neural Networks Streamlit App

## What this app does

This app is an interactive version of the notebook for D604 Task 1. It includes:

- Data loading (`images.npy` and `labels.csv`)
- Exploratory analysis (class distribution and sample images)
- Data augmentation preview
- CNN model training and evaluation
- Confusion matrix and loss visualization
- In-app display of your task documentation from the `.docx` file

## Files

- Task 1 app: `streamlit_app.py`
- Task 2 app: `streamlit_app_task2.py`
- Documentation source: `D604 Task 1 Neural Networks.docx`
- Documentation source: `D604 Task 2 Sentiment Analysis Using Neural Networks.docx`

## Setup

Recommended Python version: **3.10 or 3.11**

From the workspace root, create and activate a virtual environment, then install pinned dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r D604/requirements.txt
```

## Run

From the workspace root:

```bash
streamlit run D604/streamlit_app.py
```

For Task 2 sentiment analysis app:

```bash
streamlit run D604/streamlit_app_task2.py
```

## Data input options in app

Use either:

1. **Upload files** in the sidebar (`images.npy` and `labels.csv`), or
2. **Use local files** by entering file paths in the sidebar.

If local files are not found in the `D604` folder, use upload mode.

Task 2 expects `amazon_cells_labelled.txt` (tab-separated with review and sentiment label).

## Output

After training, the app saves the trained model to:

- `D604/d604_task1_cnn.keras`
- `D604/d604_task2_lstm.keras`

## Troubleshooting

- If TensorFlow fails to install, confirm you are using Python 3.10 or 3.11.
- If the app cannot find data files, switch to **Upload files** and provide `images.npy` and `labels.csv` manually.
- If Streamlit command is not recognized, run with:

```bash
python -m streamlit run D604/streamlit_app.py
```
