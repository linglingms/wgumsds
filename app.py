from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


st.set_page_config(page_title="D599 Market Basket Analysis", layout="wide")

st.title("D599 Task 3 Market Basket Analysis")
st.write("Interactive Streamlit app based on your notebook workflow.")


def extract_docx_text(docx_bytes):
    try:
        with zipfile.ZipFile(docx_bytes) as docx_zip:
            xml_content = docx_zip.read("word/document.xml")
        root = ET.fromstring(xml_content)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            line = "".join(texts).strip()
            if line:
                paragraphs.append(line)

        return "\n\n".join(paragraphs)
    except Exception:
        return ""


@st.cache_data
def load_source_data(uploaded_file, fallback_name):
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        return pd.read_csv(uploaded_file)

    if fallback_name.lower().endswith(".xlsx"):
        return pd.read_excel(fallback_name, engine="openpyxl")
    return pd.read_csv(fallback_name)


def preprocess_data(df):
    prepared = df.copy()
    prepared.columns = prepared.columns.str.lower().str.strip()

    order_priority_order = {"Low": 1, "Medium": 2, "High": 3}
    customer_satisfaction_order = {
        "Prefer to not respond": 0,
        "Dissatisfied": 1,
        "Very dissatisfied": 2,
        "Satisfied": 3,
        "Very Satisfied": 4,
    }

    if "orderpriority" in prepared.columns:
        prepared["orderpriority"] = prepared["orderpriority"].map(order_priority_order)

    if "customerordersatisfaction" in prepared.columns:
        prepared["customerordersatisfaction"] = prepared[
            "customerordersatisfaction"
        ].map(customer_satisfaction_order)

    if "paymentmethod" in prepared.columns:
        prepared = pd.get_dummies(prepared, columns=["paymentmethod"])

    return prepared


def build_transactions(df):
    if "orderid" not in df.columns or "productname" not in df.columns:
        raise ValueError("Required columns missing: 'orderid' and/or 'productname'.")

    transactional_df = df.groupby("orderid")["productname"].apply(list).reset_index()
    transactional_df.columns = ["orderid", "items"]
    transactions = transactional_df["items"].tolist()
    return transactional_df, transactions


def run_apriori_rules(transactions, min_support, min_lift):
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.columns_)

    frequent_itemsets = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=min_lift,
    )

    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

    return frequent_itemsets, rules


st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (.xlsx or .csv)",
    type=["xlsx", "csv"],
)

uploaded_doc = st.sidebar.file_uploader(
    "Upload documentation (.docx)",
    type=["docx"],
)

default_file = st.sidebar.selectbox(
    "Or use local file",
    options=["Megastore_Dataset_Task_3 3.xlsx", "Megastore_Dataset_Task_3 3.csv"],
)

min_support = st.sidebar.slider("Min support", min_value=0.001, max_value=0.20, value=0.01, step=0.001)
min_lift = st.sidebar.slider("Min lift", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
top_n = st.sidebar.slider("Top rules to display", min_value=3, max_value=30, value=10, step=1)

doc_text = ""
doc_file_name = "D599 Task 3 Market Basket Analysis.docx"
doc_bytes = None

if uploaded_doc is not None:
    doc_bytes = uploaded_doc.getvalue()
    doc_text = extract_docx_text(uploaded_doc)
    doc_file_name = uploaded_doc.name
else:
    local_doc_path = Path("D599 Task 3 Market Basket Analysis.docx")
    if local_doc_path.exists():
        doc_bytes = local_doc_path.read_bytes()
        doc_text = extract_docx_text(local_doc_path)

try:
    source_df = load_source_data(uploaded_file, default_file)
except FileNotFoundError:
    st.error("Dataset file not found. Upload a file or place the selected default file in this folder.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

st.subheader("1) Raw Data Preview")
st.write(f"Rows: {source_df.shape[0]} | Columns: {source_df.shape[1]}")
st.dataframe(source_df.head(10), use_container_width=True)

cleaned_df = preprocess_data(source_df)

st.subheader("2) Cleaned/Encoded Data")
st.write(f"Rows: {cleaned_df.shape[0]} | Columns: {cleaned_df.shape[1]}")
st.dataframe(cleaned_df.head(10), use_container_width=True)

try:
    transactional_df, transactions = build_transactions(source_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("3) Transactional Dataset")
st.write(f"Transactions: {len(transactional_df)}")
st.dataframe(transactional_df.head(10), use_container_width=True)

frequent_itemsets, rules_df = run_apriori_rules(
    transactions,
    min_support=min_support,
    min_lift=min_lift,
)

st.subheader("4) Frequent Itemsets")
st.write(f"Itemsets found: {len(frequent_itemsets)}")
st.dataframe(frequent_itemsets.head(20), use_container_width=True)

st.subheader("5) Association Rules")
if rules_df.empty:
    st.warning("No rules found for the selected thresholds. Try lowering min support or min lift.")
else:
    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules_sorted = rules_df.sort_values(by="lift", ascending=False)
    st.dataframe(rules_sorted[display_cols].head(top_n), use_container_width=True)

    st.subheader("Top 3 Relevant Rules (by lift)")
    st.dataframe(rules_sorted[display_cols].head(3), use_container_width=True)

st.subheader("6) Download Outputs")

cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
transaction_csv = transactional_df.to_csv(index=False).encode("utf-8")
rules_csv = rules_df.to_csv(index=False).encode("utf-8") if not rules_df.empty else b""

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        "Download cleaned data",
        data=cleaned_csv,
        file_name="cleaned_market_dataset.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Download transactional data",
        data=transaction_csv,
        file_name="transactional_dataset.csv",
        mime="text/csv",
    )
with col3:
    st.download_button(
        "Download association rules",
        data=rules_csv,
        file_name="association_rules.csv",
        mime="text/csv",
        disabled=rules_df.empty,
    )

st.subheader("7) Project Documentation")
if doc_text:
    with st.expander("View D599 documentation", expanded=False):
        st.text_area("Documentation content", value=doc_text, height=350)
    if doc_bytes is not None:
        st.download_button(
            "Download documentation (.docx)",
            data=doc_bytes,
            file_name=doc_file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
else:
    st.info("Documentation file not found. Upload a .docx file from the sidebar to display it here.")

st.caption("Built with Streamlit from D599 Task 3 notebook.")
