"""AI Smart Crop Advisor and Plant Disease Detection System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
CROP_DATASET = ROOT / "crop_recommendation_dataset_realistic.csv"
MODEL_PATH = ROOT / "random_forest_crop_model.pkl"
COLUMNS_PATH = ROOT / "model_columns.pkl"
IMAGE_DATASET = ROOT / "Dataset"
HISTORY_PATH = ROOT / "prediction_history.csv"
DISEASE_MODEL_CANDIDATES = (
    ROOT / "disease_detection_model.keras",
    ROOT / "disease_detection_model.h5",
    ROOT / "models" / "disease_detection_model.h5",
    ROOT / "models" / "plant_disease_model.keras",
)

st.set_page_config(
    page_title="AgriVision | Smart Crop Advisor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
        :root { --green:#1f6b4f; --deep:#123d2d; --lime:#b9d98b; --cream:#f7f5ed; --ink:#17332a; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: var(--cream); color: var(--ink); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#123d2d 0%,#1f6b4f 100%); }
        [data-testid="stSidebar"] * { color: #f7fff8; }
        .brand { padding: 10px 4px 24px; }
        .brand h1 { font-family:'Playfair Display',serif; font-size:26px; line-height:1.1; margin:0; color:white; }
        .brand p { color:#d8ead6; font-size:12px; margin:8px 0 0; }
        .hero { background:linear-gradient(120deg,#1d6248,#2e8a62); color:white; border-radius:24px; padding:34px 38px; box-shadow:0 14px 35px #174b3430; }
        .hero h1 { font-family:'Playfair Display',serif; font-size:42px; line-height:1.1; margin:0 0 12px; }
        .hero p { max-width:720px; color:#e5f5e6; font-size:16px; margin:0; }
        .eyebrow { color:#b9d98b; font-size:12px; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; }
        .card { background:white; border:1px solid #e7e5da; border-radius:18px; padding:22px; box-shadow:0 7px 20px #244b3710; margin-bottom:14px; }
        .metric { background:white; border-radius:16px; padding:20px; border-left:5px solid var(--lime); box-shadow:0 5px 16px #244b3710; }
        .metric .value { color:var(--deep); font-size:28px; font-weight:700; }
        .metric .label { color:#688074; font-size:13px; margin-top:3px; }
        .result { background:#eaf5e6; border:1px solid #b9d98b; border-radius:18px; padding:24px; }
        .result h2 { color:#174f38; margin:0 0 4px; }
        .workflow { text-align:center; background:white; border-radius:16px; padding:20px 10px; border:1px solid #e7e5da; }
        .workflow .node { background:#eef6e9; border-radius:12px; padding:13px 8px; min-height:56px; color:#174f38; font-weight:600; }
        div.stButton > button { border-radius:10px; font-weight:600; border:0; padding:10px 18px; }
        div.stButton > button[kind="primary"] { background:#b9d98b; color:#153c2b; }
        h1,h2,h3 { color:var(--deep); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_crop_data() -> pd.DataFrame:
    if not CROP_DATASET.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CROP_DATASET)
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Crop dataset could not be read: {exc}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_crop_artifacts() -> tuple[Any | None, list[str], str | None]:
    if not MODEL_PATH.exists():
        return None, [], "Crop model file was not found."
    try:
        model = joblib.load(MODEL_PATH)
        columns = joblib.load(COLUMNS_PATH) if COLUMNS_PATH.exists() else []
        return model, list(columns), None
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        return None, [], f"Crop model could not be loaded: {exc}"


def available_disease_classes() -> list[str]:
    labels: list[str] = []
    for candidate in (ROOT / "class_names.txt", ROOT / "data_processed" / "disease_images" / "class_names.txt"):
        if candidate.exists():
            labels = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            if labels:
                return labels
    for candidate in (ROOT / "class_names.json", ROOT / "models" / "class_names.json"):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                labels = list(data.values()) if isinstance(data, dict) else list(data)
                if labels:
                    return [str(item) for item in labels]
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    if IMAGE_DATASET.exists():
        return sorted(path.name for path in IMAGE_DATASET.iterdir() if path.is_dir())
    return []


@st.cache_resource(show_spinner=False)
def load_disease_model() -> tuple[Any | None, str | None]:
    model_path = next((path for path in DISEASE_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        return None, None
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path), str(model_path)
    except (OSError, ValueError, ImportError, ModuleNotFoundError) as exc:
        return None, f"Model exists but could not be loaded: {exc}"


def model_input_shape(model: Any) -> tuple[int, int]:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) < 3 or shape[1] is None or shape[2] is None:
        return 224, 224
    return int(shape[1]), int(shape[2])


def disease_prediction(model: Any, image: Image.Image, labels: list[str]) -> tuple[str, float | None, list[tuple[str, float]]]:
    height, width = model_input_shape(model)
    prepared = image.convert("RGB").resize((width, height))
    array = np.asarray(prepared, dtype=np.float32)[None, ...]
    first_layer = getattr(getattr(model, "layers", [None])[0], "__class__", type(None)).__name__
    if first_layer != "Rescaling":
        array /= 255.0
    raw = np.asarray(model.predict(array, verbose=0))
    scores = raw[0] if raw.ndim > 1 else raw
    if len(scores) == 0:
        raise ValueError("The disease model returned no classes.")
    if len(labels) != len(scores):
        raise ValueError(
            f"Model returns {len(scores)} classes but class_names.txt contains {len(labels)} labels."
        )
    if np.all(scores >= 0) and np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        probabilities = scores / scores.sum()
    else:
        probabilities = np.exp(scores - np.max(scores))
        probabilities = probabilities / probabilities.sum()
    ranked = np.argsort(probabilities)[::-1][:3]
    top = [(labels[index] if index < len(labels) else f"Class {index}", float(probabilities[index])) for index in ranked]
    return top[0][0], top[0][1], top


def read_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "filename", "prediction", "confidence"])
    try:
        return pd.read_csv(HISTORY_PATH)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame(columns=["timestamp", "filename", "prediction", "confidence"])


def save_history(filename: str, prediction: str, confidence: float | None) -> None:
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
    }])
    history = pd.concat([read_history(), row], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)


def metric_card(label: str, value: str) -> None:
    st.markdown(f'<div class="metric"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)


def crop_page(data: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Decision support</div><h1>Crop Recommendation</h1>', unsafe_allow_html=True)
    st.write("Enter the current farm conditions. Inputs mirror the features used by the saved crop model.")
    model, model_columns, error = load_crop_artifacts()
    if error:
        st.warning(error)
    input_features = [column for column in data.columns if column != "Crop"]
    if not input_features:
        st.info("Add the trained crop model and its feature-column artifact to enable recommendations.")
        return
    values: dict[str, Any] = {}
    categorical = {"State", "District", "Season"}
    with st.form("crop_form"):
        st.markdown("#### Farm and climate profile")
        columns = st.columns(3)
        for index, feature in enumerate(input_features):
            slot = columns[index % 3]
            with slot:
                if feature in categorical and feature in data.columns:
                    options = sorted(data[feature].dropna().astype(str).unique().tolist())
                    values[feature] = st.selectbox(feature.replace("_", " "), options)
                elif feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                    series = pd.to_numeric(data[feature], errors="coerce").dropna()
                    minimum = float(series.min()) if not series.empty else 0.0
                    maximum = float(series.max()) if not series.empty else 100.0
                    default = float(series.median()) if not series.empty else minimum
                    values[feature] = st.number_input(feature.replace("_", " "), min_value=minimum, max_value=maximum, value=default, step=0.01)
                else:
                    values[feature] = st.text_input(feature.replace("_", " "), placeholder="Required")
        submitted = st.form_submit_button("🌱 Generate recommendation", type="primary", use_container_width=True)
    if not submitted:
        return
    missing = [name for name, value in values.items() if value is None or str(value).strip() == ""]
    if missing:
        st.error(f"Please complete: {', '.join(missing)}")
        return
    if model is None:
        st.info("The form is ready, but a compatible saved crop model is required to predict.")
        return
    raw = pd.DataFrame([values])
    encoded = pd.get_dummies(raw, columns=[name for name in categorical if name in raw.columns], drop_first=True)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    try:
        prediction = model.predict(encoded)[0]
        label = str(prediction)
        st.markdown(f'<div class="result"><div class="eyebrow">Model recommendation</div><h2>{label}</h2><p>Based on the farm profile you entered.</p></div>', unsafe_allow_html=True)
        report = pd.DataFrame([{"predicted_crop": label, **values}])
        st.download_button("Download recommendation report", report.to_csv(index=False).encode("utf-8"), "crop_recommendation.csv", "text/csv")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(encoded)[0]
            classes = getattr(model, "classes_", [])
            top = np.argsort(probabilities)[::-1][:3]
            st.markdown("#### Top model-supported recommendations")
            for rank, index in enumerate(top, 1):
                st.write(f"**{rank}. {classes[index]}** — {probabilities[index]:.1%}")
    except (ValueError, TypeError, AttributeError) as exc:
        st.error(f"Prediction failed because the saved model rejected the input schema: {exc}")


def disease_page() -> None:
    st.markdown('<div class="eyebrow">Visual diagnosis</div><h1>Plant Disease Detection</h1>', unsafe_allow_html=True)
    st.write("Upload a clear image of one leaf. The app will only report a disease when a trained image model is available.")
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], help="Use a well-lit image with one leaf in focus.")
    if not uploaded:
        st.info("No image selected yet.")
        return
    try:
        image = Image.open(uploaded)
        image.verify()
        uploaded.seek(0)
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file is not a readable JPG, JPEG, or PNG image.")
        return
    left, right = st.columns([1.1, 1])
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)
    with right:
        st.markdown(f'<div class="card"><b>Image details</b><br>{image.width} × {image.height} px<br>{uploaded.type or "image"}</div>', unsafe_allow_html=True)
        model, model_info = load_disease_model()
        labels = available_disease_classes()
        if model is None:
            if model_info:
                st.error(model_info)
            else:
                st.warning("A trained disease model was not found. Add disease_detection_model.h5 or .keras to connect detection.")
            if labels:
                st.caption(f"Dataset metadata contains {len(labels)} class folders; these are not predictions.")
            return
        if st.button("🔍 Detect Plant Disease", type="primary", use_container_width=True):
            with st.spinner("Preprocessing image and running the model..."):
                try:
                    prediction, confidence, top = disease_prediction(model, image, labels)
                    st.markdown(f'<div class="result"><div class="eyebrow">Prediction</div><h2>{prediction}</h2><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                    if confidence is not None and confidence < 0.60:
                        st.warning("The model is uncertain about this image. Use a clear, well-lit leaf photo and confirm the result with an agricultural expert.")
                    st.markdown("#### Top classes")
                    for name, score in top:
                        st.write(f"**{name}** — {score:.1%}")
                    save_history(uploaded.name, prediction, confidence)
                    st.success("Prediction saved to local history.")
                except (ValueError, TypeError, RuntimeError) as exc:
                    st.error(f"Disease prediction could not be completed: {exc}")
    st.caption("Disclaimer: predictions are model-based estimates. Consult an agricultural expert for uncertain or severe cases.")


def dashboard(data: pd.DataFrame) -> None:
    model, _, _ = load_crop_artifacts()
    history = read_history()
    st.markdown('<div class="hero"><div class="eyebrow">AgriVision platform</div><h1>Smarter decisions for healthier harvests.</h1><p>Welcome, farmer. Use data-driven crop guidance and transparent plant health tools to plan with confidence.</p></div>', unsafe_allow_html=True)
    st.write("")
    cards = st.columns(4)
    with cards[0]:
        metric_card("Available crops", str(data["Crop"].nunique()) if "Crop" in data else "—")
    with cards[1]:
        metric_card("Disease classes in dataset", str(len(available_disease_classes())) or "—")
    with cards[2]:
        metric_card("Saved predictions", str(len(history)))
    with cards[3]:
        metric_card("Crop model", "Ready" if model is not None else "Unavailable")
    st.write("")
    a, b = st.columns(2)
    with a:
        if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
            st.session_state.page = "Crop Recommendation"
            st.rerun()
    with b:
        if st.button("🍃 Detect Plant Disease", use_container_width=True):
            st.session_state.page = "Plant Disease Detection"
            st.rerun()
    st.markdown("#### How it works")
    flow = st.columns(7)
    steps = ["Farm details", "→", "Crop model", "→", "Suitable crop", "—", "Leaf image → preprocessing → disease model → result"]
    for slot, step in zip(flow, steps):
        with slot:
            st.markdown(f'<div class="workflow"><div class="node">{step}</div></div>', unsafe_allow_html=True)


def analytics_page(data: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Evidence and trends</div><h1>Crop Analytics</h1>', unsafe_allow_html=True)
    if data.empty:
        st.info("Load the agricultural CSV to see analytics.")
        return
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(data["Crop"].value_counts().reset_index(), x="Crop", y="count", title="Crop distribution", color_discrete_sequence=["#2e8a62"]), use_container_width=True)
    with right:
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            selected = st.selectbox("Feature distribution", numeric)
            st.plotly_chart(px.histogram(data, x=selected, title=f"{selected} distribution", color_discrete_sequence=["#b9d98b"]), use_container_width=True)
    st.caption("Charts above describe the loaded training/reference dataset. They are not real-time prediction accuracy.")


def history_page() -> None:
    st.markdown('<div class="eyebrow">Local records</div><h1>Prediction History</h1>', unsafe_allow_html=True)
    history = read_history()
    if history.empty:
        st.info("No disease predictions have been saved yet.")
        return
    query = st.text_input("Search filename or prediction")
    shown = history[history.astype(str).apply(lambda row: row.str.contains(query, case=False, na=False).any(), axis=1)] if query else history
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("Download history CSV", shown.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv")


def about_page() -> None:
    st.markdown('<div class="eyebrow">Project documentation</div><h1>About the project</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card"><h3>Overview</h3><p>AI Smart Crop Advisor and Plant Disease Detection System is a Streamlit decision-support frontend for agricultural machine-learning models.</p>
        <h3>Architecture</h3><p><b>Crop:</b> User input → validation → saved preprocessing schema → Random Forest model → recommendation.</p>
        <p><b>Disease:</b> Leaf upload → image validation → model input shape and normalization → trained classifier → result and guidance.</p>
        <h3>Technology</h3><p>Python, Streamlit, Pandas, NumPy, Plotly, Joblib, Pillow, scikit-learn, and optional TensorFlow/Keras.</p></div>
        """,
        unsafe_allow_html=True,
    )
    st.info("The disease model is optional in this installation. Add a compatible .h5 or .keras model and class_names.txt/class_names.json to enable it.")


def main() -> None:
    inject_css()
    data = load_crop_data()
    if "page" not in st.session_state:
        st.session_state.page = "Home Dashboard"
    with st.sidebar:
        st.markdown('<div class="brand"><h1>AgriVision</h1><p>AI-powered crop intelligence</p></div>', unsafe_allow_html=True)
        pages = ["Home Dashboard", "Crop Recommendation", "Plant Disease Detection", "Crop Analytics", "Prediction History", "About Project"]
        st.session_state.page = st.radio("Navigate", pages, index=pages.index(st.session_state.page))
        st.divider()
        st.caption("Built for practical, transparent farm decisions.")
    page = st.session_state.page
    if page == "Home Dashboard":
        dashboard(data)
    elif page == "Crop Recommendation":
        crop_page(data)
    elif page == "Plant Disease Detection":
        disease_page()
    elif page == "Crop Analytics":
        analytics_page(data)
    elif page == "Prediction History":
        history_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
