from pathlib import Path
import io
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


st.set_page_config(page_title="D604 Task 1 - Neural Networks App", layout="wide")


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
def load_dataset(images_bytes: bytes, labels_bytes: bytes):
    images = np.load(io.BytesIO(images_bytes))
    labels_df = pd.read_csv(io.BytesIO(labels_bytes))
    if "Label" in labels_df.columns:
        labels = labels_df["Label"].astype(str).values
    else:
        labels = labels_df.iloc[:, 0].astype(str).values
    return images, labels, labels_df


def build_cnn_model(input_shape, num_classes):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def render_overview(images, labels):
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Image array shape: {images.shape}")
        st.write(f"Label count: {len(labels)}")

    with col2:
        class_counts = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        class_counts.plot(kind="bar", ax=ax)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Sample Images")
    max_samples = min(5, len(images))
    sample_cols = st.columns(max_samples)
    for i in range(max_samples):
        with sample_cols[i]:
            st.image(images[i], caption=f"Label: {labels[i]}", use_container_width=True)


def render_augmentation(images):
    st.subheader("Data Augmentation Preview")
    if not TF_AVAILABLE:
        st.warning("TensorFlow is not installed, so augmentation preview is unavailable.")
        return

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    sample_image = images[0].reshape((1,) + images[0].shape)
    augmented_images = datagen.flow(sample_image, batch_size=1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        aug = next(augmented_images)[0].astype("uint8")
        axes[i].imshow(aug)
        axes[i].axis("off")
    st.pyplot(fig)


def render_training(images, labels):
    st.subheader("Train, Validate, and Evaluate CNN")

    if not TF_AVAILABLE:
        st.error("TensorFlow is required to train the model. Install it and rerun the app.")
        return

    with st.sidebar:
        st.markdown("### Training Controls")
        test_size = st.slider("Test split (%)", 10, 30, 15)
        val_size = st.slider("Validation split from remaining (%)", 10, 30, 15)
        epochs = st.slider("Epochs", 5, 50, 20)
        batch_size = st.select_slider("Batch size", options=[16, 32, 64], value=32)

    images_norm = images.astype("float32") / 255.0

    X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
        images_norm,
        labels,
        test_size=(test_size + val_size) / 100,
        random_state=42,
        stratify=labels,
    )

    adjusted_val = val_size / (test_size + val_size)
    X_val, X_test, y_val_raw, y_test_raw = train_test_split(
        X_temp,
        y_temp_raw,
        test_size=1 - adjusted_val,
        random_state=42,
        stratify=y_temp_raw,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    num_classes = len(label_encoder.classes_)
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_val_encoded = to_categorical(y_val, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    st.write(
        {
            "Train Shape": X_train.shape,
            "Validation Shape": X_val.shape,
            "Test Shape": X_test.shape,
            "Classes": list(label_encoder.classes_),
        }
    )

    if st.button("Train Model", type="primary"):
        model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        with st.spinner("Training model..."):
            history = model.fit(
                X_train,
                y_train_encoded,
                validation_data=(X_val, y_val_encoded),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0,
            )

        train_loss, train_acc = model.evaluate(X_train, y_train_encoded, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val_encoded, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=0)

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

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history["loss"], label="Training Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        st.pyplot(fig)

        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test_encoded, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted Labels")
        ax_cm.set_ylabel("True Labels")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        model_path = Path(__file__).parent / "d604_task1_cnn.keras"
        model.save(model_path)
        st.info(f"Saved trained model to: {model_path}")


def main():
    st.title("D604 Task 1 - Neural Networks (Streamlit)")
    st.caption("Interactive app version of your notebook, including your documentation.")

    default_doc_path = Path(__file__).parent / "D604 Task 1 Neural Networks.docx"
    default_images_path = Path(__file__).parent / "images.npy"
    default_labels_path = Path(__file__).parent / "labels.csv"

    tab_data, tab_train, tab_docs = st.tabs(["Data & EDA", "Modeling", "Documentation"])

    with st.sidebar:
        st.markdown("### Dataset Source")
        source = st.radio("Choose input method", ["Upload files", "Use local files"], index=0)
        uploaded_images = st.file_uploader("Upload images.npy", type=["npy"]) if source == "Upload files" else None
        uploaded_labels = st.file_uploader("Upload labels.csv", type=["csv"]) if source == "Upload files" else None
        local_images = st.text_input("Local images path", value=str(default_images_path)) if source == "Use local files" else None
        local_labels = st.text_input("Local labels path", value=str(default_labels_path)) if source == "Use local files" else None

    images = None
    labels = None

    try:
        if source == "Upload files" and uploaded_images is not None and uploaded_labels is not None:
            images, labels, _ = load_dataset(uploaded_images.read(), uploaded_labels.read())
        elif source == "Use local files" and Path(local_images).exists() and Path(local_labels).exists():
            with open(local_images, "rb") as f_img, open(local_labels, "rb") as f_lbl:
                images, labels, _ = load_dataset(f_img.read(), f_lbl.read())
    except Exception as ex:
        st.error(f"Could not load dataset: {ex}")

    with tab_data:
        if images is None or labels is None:
            st.warning("Provide both dataset files (`images.npy` and `labels.csv`) to view EDA.")
        else:
            render_overview(images, labels)
            render_augmentation(images)

    with tab_train:
        if images is None or labels is None:
            st.warning("Load dataset first to train and evaluate the model.")
        else:
            render_training(images, labels)

    with tab_docs:
        st.subheader("Task Documentation")
        doc_path = st.text_input("Documentation (.docx) path", value=str(default_doc_path))
        documentation_text = extract_docx_text(doc_path)
        st.text_area("Extracted Documentation", value=documentation_text, height=500)


if __name__ == "__main__":
    main()
