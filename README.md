# D599 Task 3 - Market Basket Analysis (Streamlit App)

This project converts your notebook workflow into a deployable Streamlit app.

## App File

- `app.py`

## What the App Does

- Loads `Megastore_Dataset_Task_3 3.xlsx` or `Megastore_Dataset_Task_3 3.csv` (or an uploaded file)
- Applies the same preprocessing from the notebook:
  - normalizes column names
  - ordinal encoding for `orderpriority` and `customerordersatisfaction`
  - one-hot encoding for `paymentmethod`
- Builds transaction baskets from `orderid` and `productname`
- Runs Apriori + association rules
- Shows top rules by lift and lets you download outputs

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

## Streamlit Community Cloud Deployment

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select your repository and set the main file path to `app.py`.
4. Deploy.

## Notes

- If default local files are not present in deployment, upload your dataset using the app sidebar uploader.
- Outputs can be downloaded directly from the app as CSV files.
