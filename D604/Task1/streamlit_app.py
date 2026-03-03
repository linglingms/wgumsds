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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as ex:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(ex)


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


def create_demo_dataset(samples_per_class: int = 10, image_size: int = 64, num_classes: int = 12):
    rng = np.random.default_rng(42)
    total = samples_per_class * num_classes
    images = rng.integers(0, 256, size=(total, image_size, image_size, 3), dtype=np.uint8)
    labels = np.array([f"class_{i}" for i in range(num_classes) for _ in range(samples_per_class)])
    return images, labels


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
        sample = images[0]
        h, w = sample.shape[:2]

        flip_h = np.fliplr(sample)
        rot_90 = np.rot90(sample)
        brighten = np.clip(sample.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)

        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        crop = sample[top : top + crop_h, left : left + crop_w]
        zoom_in = np.zeros_like(sample)
        y_idx = np.linspace(0, crop_h - 1, h).astype(int)
        x_idx = np.linspace(0, crop_w - 1, w).astype(int)
        zoom_in = crop[np.ix_(y_idx, x_idx)]

        fallback_images = [sample, flip_h, rot_90, brighten, zoom_in]
        fallback_titles = ["Original", "Flip Horizontal", "Rotate 90°", "Brighten", "Zoom In"]

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axes[i].imshow(fallback_images[i])
            axes[i].set_title(fallback_titles[i], fontsize=9)
            axes[i].axis("off")
        st.pyplot(fig)
        st.caption("Using built-in augmentation preview fallback (TensorFlow not required).")
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
        st.info("Running fallback training mode (scikit-learn) for this environment.")
        with st.expander("TensorFlow details (optional)"):
            st.caption(f"TensorFlow import not available: {TF_IMPORT_ERROR}")

        with st.sidebar:
            st.markdown("### Fallback Training Controls")
            test_size = st.slider("Test split (%)", 10, 30, 15, key="fb_test")
            val_size = st.slider("Validation split from remaining (%)", 10, 30, 15, key="fb_val")
            max_iter = st.slider("Max iterations", 50, 300, 100, key="fb_iter")

        images_norm = images.astype("float32") / 255.0
        X_flat = images_norm.reshape(images_norm.shape[0], -1)

        X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
            X_flat,
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

        st.write(
            {
                "Train Shape": X_train.shape,
                "Validation Shape": X_val.shape,
                "Test Shape": X_test.shape,
                "Classes": list(label_encoder.classes_),
            }
        )

        if st.button("Train Fallback Model", type="primary"):
            model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=max_iter, random_state=42)
            with st.spinner("Training fallback model..."):
                model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            test_acc = model.score(X_test, y_test)

            st.success("Fallback training complete.")
            st.write(
                {
                    "Training Accuracy": round(float(train_acc), 4),
                    "Validation Accuracy": round(float(val_acc), 4),
                    "Test Accuracy": round(float(test_acc), 4),
                }
            )

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
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
            ax_cm.set_title("Confusion Matrix (Fallback Model)")
            st.pyplot(fig_cm)

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
    default_local_available = default_images_path.exists() and default_labels_path.exists()

    tab_data, tab_train, tab_docs = st.tabs(["Data & EDA", "Modeling", "Documentation"])

    with st.sidebar:
        st.markdown("### Dataset Source")
        source = st.radio(
            "Choose input method",
            ["Upload files", "Use local files"],
            index=1 if default_local_available else 0,
        )
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

    if images is None or labels is None:
        if source == "Use local files":
            missing = []
            if not Path(local_images).exists():
                missing.append(f"images file not found: {local_images}")
            if not Path(local_labels).exists():
                missing.append(f"labels file not found: {local_labels}")
            if missing:
                st.error("Local files could not be loaded:\n- " + "\n- ".join(missing))
                st.info("On Streamlit Cloud, local workspace files are not available unless committed. Use Upload files.")
                st.info("Upload `images.npy` and `labels.csv`, or use Demo Data below.")
        else:
            st.info("Upload both `images.npy` and `labels.csv` to load data, or use Demo Data below.")

        if st.button("Use Demo Data", type="secondary"):
            images, labels = create_demo_dataset()
            st.success("Loaded demo dataset so you can use the app immediately.")

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
