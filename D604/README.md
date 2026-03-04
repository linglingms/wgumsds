# D604 Streamlit Apps (Task 1 and Task 2)

## Separate apps

Task 1 and Task 2 are separate Streamlit apps with separate entrypoints:

- Task 1 main file: `D604/Task1/task1_app.py`
- Task 2 main file: `D604/Task2/task2_app.py`

## Task 1 app (Neural Networks)

This app is an interactive version of the notebook for D604 Task 1. It includes:

- Data loading (`images.npy` and `labels.csv`)
- Exploratory analysis (class distribution and sample images)
- Data augmentation preview
- CNN model training and evaluation
- Confusion matrix and loss visualization
- In-app display of your task documentation from the `.docx` file

## Files

- Task 1 app logic: `Task1/streamlit_app.py`
- Task 1 app entrypoint: `Task1/task1_app.py`
- Task 2 app logic: `Task2/streamlit_app_task2.py`
- Task 2 app entrypoint: `Task2/task2_app.py`
- Task 1 documentation source: `Task1/D604 Task 1 Neural Networks.docx`
- Task 2 documentation source: `Task2/D604 Task 2 Sentiment Analysis Using Neural Networks.docx`
- Task 2 dataset source: `Task2/amazon_cells_labelled.txt`

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

From the workspace root (Task 1):

```bash
streamlit run D604/Task1/task1_app.py
```

Task 2 sentiment analysis app:

```bash
streamlit run D604/Task2/task2_app.py
```

## Data input options in app

Use either:

1. **Upload files** in the sidebar (`images.npy` and `labels.csv`), or
2. **Use local files** by entering file paths in the sidebar.

If local files are not found in the task folder, use upload mode.

For Task 1 on Streamlit Cloud, local files such as `images.npy` and `labels.csv` may not exist unless committed to the repository. If they are missing, use:

- **Upload files** (`images.npy` + `labels.csv`), or
- **Use Demo Data** button in the app.

Task 2 expects `amazon_cells_labelled.txt` in `D604/Task2` (tab-separated with review and sentiment label).

## Output

After training, the app saves the trained model to:

- `D604/Task1/d604_task1_cnn.keras`
- `D604/Task2/d604_task2_lstm.keras`

## Troubleshooting

- If TensorFlow fails to install, confirm you are using Python 3.10 or 3.11.
- If the app cannot find data files, switch to **Upload files** and provide `images.npy` and `labels.csv` manually.
- If Task 1 shows missing local files in Cloud, click **Use Demo Data** to run the app immediately.
- If Streamlit command is not recognized, run with:

```bash
python -m streamlit run D604/Task1/task1_app.py
```
