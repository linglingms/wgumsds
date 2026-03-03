from pathlib import Path
import io
import re
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf

    EarlyStopping = tf.keras.callbacks.EarlyStopping
    Dense = tf.keras.layers.Dense
    Embedding = tf.keras.layers.Embedding
    LSTM = tf.keras.layers.LSTM
    Sequential = tf.keras.models.Sequential
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    Tokenizer = tf.keras.preprocessing.text.Tokenizer

    TF_AVAILABLE = True
except Exception:
    tf = None
    EarlyStopping = None
    Dense = None
    Embedding = None
    LSTM = None
    Sequential = None
    pad_sequences = None
    Tokenizer = None
    TF_AVAILABLE = False


st.set_page_config(page_title="D604 Task 2 - Sentiment LSTM App", layout="wide")


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


@st.cache_data
def load_dataset_from_tsv(tsv_content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(tsv_content), names=["review", "sentiment"], sep="\t")
    return df


@st.cache_data
def load_dataset_from_local(tsv_path: str) -> pd.DataFrame:
    return pd.read_csv(tsv_path, names=["review", "sentiment"], sep="\t")


def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def build_lstm_model(vocab_size: int, max_seq_len: int, embedding_dim: int = 100) -> Any:
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
            LSTM(64),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def render_eda(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    st.write(df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Missing Reviews", int(df["review"].isnull().sum()))
    with col3:
        st.metric("Missing Sentiments", int(df["sentiment"].isnull().sum()))

    df = df.copy()
    df["review"] = df["review"].astype(str)
    df["unusual_chars"] = df["review"].apply(
        lambda x: "".join([char for char in x if not char.isalnum() and not char.isspace()])
    )
    st.write("Unusual character patterns:")
    st.dataframe(df["unusual_chars"].value_counts().head(10))

    st.write("Sentiment distribution:")
    fig, ax = plt.subplots(figsize=(6, 3))
    pd.Series(df["sentiment"]).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Class Distribution")
    st.pyplot(fig)


def prepare_data(df: pd.DataFrame, max_vocab_size: int, max_seq_limit: int):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for tokenization and sequence padding.")

    work_df = df.copy()
    work_df["review"] = work_df["review"].astype(str)
    work_df["clean_review"] = work_df["review"].apply(clean_text)

    work_df["review_length"] = work_df["clean_review"].apply(lambda x: len(x.split()))
    quantile_seq_len = int(work_df["review_length"].quantile(0.95))
    max_seq_len = min(max_seq_limit, max(1, quantile_seq_len))

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(work_df["clean_review"])

    sequences = tokenizer.texts_to_sequences(work_df["clean_review"])
    padded = pad_sequences(sequences, maxlen=max_seq_len, padding="post", truncating="post")

    labels = pd.to_numeric(work_df["sentiment"], errors="coerce").fillna(0).astype(int).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        padded,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None,
    )

    effective_vocab = min(max_vocab_size, len(tokenizer.word_index) + 1)

    return {
        "data": work_df,
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "effective_vocab": effective_vocab,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def render_training(prepared, epochs: int, batch_size: int):
    if not TF_AVAILABLE:
        st.error("TensorFlow is required to train the Task 2 model. Install dependencies and rerun.")
        return

    st.subheader("Train LSTM Model")
    st.write(
        {
            "Train Shape": prepared["X_train"].shape,
            "Validation Shape": prepared["X_val"].shape,
            "Test Shape": prepared["X_test"].shape,
            "Effective Vocabulary": prepared["effective_vocab"],
            "Max Sequence Length": prepared["max_seq_len"],
        }
    )

    if st.button("Train Sentiment Model", type="primary"):
        model = build_lstm_model(
            vocab_size=prepared["effective_vocab"],
            max_seq_len=prepared["max_seq_len"],
            embedding_dim=100,
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        with st.spinner("Training model..."):
            history = model.fit(
                prepared["X_train"],
                prepared["y_train"],
                validation_data=(prepared["X_val"], prepared["y_val"]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0,
            )

        train_loss, train_acc = model.evaluate(prepared["X_train"], prepared["y_train"], verbose=0)
        val_loss, val_acc = model.evaluate(prepared["X_val"], prepared["y_val"], verbose=0)
        test_loss, test_acc = model.evaluate(prepared["X_test"], prepared["y_test"], verbose=0)

        st.success("Training complete.")
        st.write(
            {
                "Training Accuracy": round(float(train_acc), 4),
                "Validation Accuracy": round(float(val_acc), 4),
                "Test Accuracy": round(float(test_acc), 4),
                "Training Loss": round(float(train_loss), 4),
                "Validation Loss": round(float(val_loss), 4),
                "Test Loss": round(float(test_loss), 4),
            }
        )

        fig_acc, ax_acc = plt.subplots(figsize=(7, 3))
        ax_acc.plot(history.history["accuracy"], label="Train Accuracy")
        ax_acc.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.set_title("Training vs Validation Accuracy")
        st.pyplot(fig_acc)

        fig_loss, ax_loss = plt.subplots(figsize=(7, 3))
        ax_loss.plot(history.history["loss"], label="Train Loss")
        ax_loss.plot(history.history["val_loss"], label="Validation Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.set_title("Training vs Validation Loss")
        st.pyplot(fig_loss)

        model_path = Path(__file__).parent / "d604_task2_lstm.keras"
        model.save(model_path)
        st.info(f"Saved trained model to: {model_path}")


def render_prepared_download(prepared):
    st.subheader("Prepared Dataset Export")
    prepared_train_df = pd.DataFrame(prepared["X_train"])
    prepared_train_df["label"] = prepared["y_train"]
    csv_data = prepared_train_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download prepared_sentiment_data.csv",
        data=csv_data,
        file_name="prepared_sentiment_data.csv",
        mime="text/csv",
    )


def render_documentation(doc_path: str):
    documentation_text = extract_docx_text(doc_path)
    if documentation_text == "Documentation file not found.":
        st.warning(documentation_text)
        return

    lines = [line.strip() for line in documentation_text.splitlines() if line.strip()]
    if not lines:
        st.warning("No documentation text found.")
        return

    section_pattern = re.compile(r"^Part\s+[IVXLC]+:\s*", re.IGNORECASE)
    question_pattern = re.compile(r"^([A-Z]\.\s+|\d+\.\s+)")

    current_section = "Documentation"
    qa_items = []
    current_question = None
    current_answer_lines = []
    current_inline_answer = ""

    def parse_question_line(line_text: str):
        prefix_match = re.match(r"^([A-Z]\.\s+|\d+\.\s+)(.*)$", line_text)
        content = prefix_match.group(2).strip() if prefix_match else line_text.strip()
        q_text = line_text
        inline_answer = ""

        if "?" in content:
            left, right = content.split("?", 1)
            q_text = f"{left.strip()}?"
            inline_answer = right.strip()
        else:
            split_sentences = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)
            if len(split_sentences) == 2 and split_sentences[1].strip():
                q_text = split_sentences[0].strip()
                inline_answer = split_sentences[1].strip()
            else:
                q_text = content

        return q_text, inline_answer

    def flush_question():
        nonlocal current_question, current_answer_lines, current_inline_answer
        if current_question:
            answer_text = "\n".join(current_answer_lines).strip()
            if not answer_text:
                answer_text = current_inline_answer.strip()
            qa_items.append(
                {
                    "section": current_section,
                    "question": current_question,
                    "answer": answer_text,
                }
            )
        current_question = None
        current_answer_lines = []
        current_inline_answer = ""

    for line in lines:
        if section_pattern.match(line):
            flush_question()
            current_section = line
            continue

        is_question = bool(question_pattern.match(line)) or line.endswith("?")
        if is_question:
            flush_question()
            current_question, current_inline_answer = parse_question_line(line)
        else:
            if current_question:
                current_answer_lines.append(line)

    flush_question()

    if not qa_items:
        st.info("Could not auto-structure this document. Showing raw text below.")
    else:
        shown_sections = set()
        for item in qa_items:
            if item["section"] not in shown_sections:
                st.markdown(f"### {item['section']}")
                shown_sections.add(item["section"])
            with st.container(border=True):
                st.markdown(f"**Question**\n\n{item['question']}")
                st.markdown(f"**Answer**\n\n{item['answer'] or '_No answer provided._'}")

    with st.expander("View raw documentation text"):
        st.text_area("Raw Documentation", value=documentation_text, height=300)


def main():
    st.title("D604 Task 2 - Sentiment Analysis with Neural Networks")
    st.caption("Interactive app version of your Task 2 notebook, including your documentation.")

    default_doc = Path(__file__).parent / "D604 Task 2 Sentiment Analysis Using Neural Networks.docx"
    default_tsv = Path(__file__).parent / "amazon_cells_labelled.txt"
    default_local_available = default_tsv.exists()

    with st.sidebar:
        st.markdown("### Data Source")
        source = st.radio("Choose input method", ["Upload file", "Use local file"], index=1 if default_local_available else 0)
        uploaded_tsv = (
            st.file_uploader("Upload amazon_cells_labelled.txt", type=["txt", "tsv"])
            if source == "Upload file"
            else None
        )
        local_tsv = (
            st.text_input("Local TSV/TXT path", value=str(default_tsv))
            if source == "Use local file"
            else None
        )

        st.markdown("### Preprocessing Controls")
        max_vocab_size = st.number_input("Max Vocabulary Size", min_value=100, max_value=50000, value=1880, step=100)
        max_seq_limit = st.number_input("Max Sequence Length Limit", min_value=10, max_value=500, value=100, step=5)

        st.markdown("### Training Controls")
        epochs = st.slider("Epochs", 5, 30, 20)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64], value=32)

    df = None
    try:
        if source == "Upload file" and uploaded_tsv is not None:
            df = load_dataset_from_tsv(uploaded_tsv.read())
        elif source == "Use local file" and Path(local_tsv).exists():
            df = load_dataset_from_local(local_tsv)
    except Exception as ex:
        st.error(f"Could not load dataset: {ex}")

    if df is None:
        if source == "Use local file":
            st.error(f"Local dataset file not found: {local_tsv}")
            st.info("Switch to Upload file and provide `amazon_cells_labelled.txt`.")
        else:
            st.info("Upload `amazon_cells_labelled.txt` to load data.")

    tab_data, tab_model, tab_docs = st.tabs(["Data & Prep", "Modeling", "Documentation"])

    if df is not None and TF_AVAILABLE:
        prepared = prepare_data(df, max_vocab_size=int(max_vocab_size), max_seq_limit=int(max_seq_limit))
    else:
        prepared = None

    with tab_data:
        if df is None:
            st.warning("Load a sentiment dataset file to continue.")
        elif not TF_AVAILABLE:
            render_eda(df)
            st.warning("TensorFlow is not available, so tokenization, padding, and training are disabled.")
        else:
            render_eda(df)
            st.write(f"95th percentile sequence length (capped): {prepared['max_seq_len']}")
            st.write("Sample padded sequence:")
            st.code(str(prepared["X_train"][0][: min(40, prepared["X_train"].shape[1])]))
            render_prepared_download(prepared)

    with tab_model:
        if not TF_AVAILABLE:
            st.warning("TensorFlow is not available in this environment, so model training is disabled.")
        elif prepared is None:
            st.warning("Prepare data first to train and evaluate the model.")
        else:
            render_training(prepared, epochs=epochs, batch_size=batch_size)

    with tab_docs:
        st.subheader("Task Documentation")
        doc_path = st.text_input("Documentation (.docx) path", value=str(default_doc))
        render_documentation(doc_path)


if __name__ == "__main__":
    main()
