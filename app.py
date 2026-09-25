from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# Load the model and validation data from the application directory.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "ann_model.pkl"
DATA_PATH = APP_DIR / "valdata.csv"

model = joblib.load(MODEL_PATH)
validation_data = pd.read_csv(DATA_PATH)

# The saved MLP was trained with these four columns, in this order.
feature_names = list(getattr(model, "feature_names_in_", []))
if not feature_names:
    feature_names = ["BMI", "Number.of.staples", "Chest.drains", "Ondansetron"]

# Keep the model's original column names internally, but use clear labels in the UI.
display_feature_names = {
    "BMI": "BMI",
    "Number.of.staples": "Number of staplers",
    "Chest.drains": "Chest drains",
    "Ondansetron": "Ondansetron",
}

required_columns = {"Outcome", *feature_names}
missing_columns = required_columns.difference(validation_data.columns)
if missing_columns:
    raise ValueError(f"Missing columns in valdata.csv: {sorted(missing_columns)}")

background_data = validation_data[feature_names].sample(
    n=min(50, len(validation_data)), random_state=42
)


def _numeric_bounds(column, default_min, default_max):
    values = pd.to_numeric(validation_data[column], errors="coerce").dropna()
    if values.empty:
        return default_min, default_max
    return float(values.min()), float(values.max())


# Streamlit user interface
st.title("CRP Predictor")

# BMI: numerical input
bmi_min, bmi_max = _numeric_bounds("BMI", 0.0, 60.0)
bmi = st.number_input(
    "Body mass index (BMI):",
    min_value=float(min(0.0, bmi_min)),
    max_value=float(max(60.0, bmi_max)),
    value=float(validation_data["BMI"].median()),
    step=0.01,
)

# Number.of.staples: numerical input (displayed as Number of staplers)
staplers_min, staplers_max = _numeric_bounds("Number.of.staples", 0, 50)
number_of_staplers = st.number_input(
    "Number of staplers:",
    min_value=int(staplers_min),
    max_value=int(staplers_max),
    value=int(round(validation_data["Number.of.staples"].median())),
    step=1,
)

# Chest.drains: categorical selection
chest_drain_options = sorted(
    pd.to_numeric(validation_data["Chest.drains"], errors="coerce")
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)
chest_drains = st.selectbox(
    "Number of chest drains:",
    options=chest_drain_options,
    format_func=lambda value: f"{value}",
)

# Ondansetron: numerical input
ondansetron_min, ondansetron_max = _numeric_bounds("Ondansetron", 0, 100)
ondansetron = st.number_input(
    "Ondansetron dose:",
    min_value=int(ondansetron_min),
    max_value=int(ondansetron_max),
    value=int(round(validation_data["Ondansetron"].median())),
    step=1,
)

# Process inputs and make predictions
feature_values = [bmi, number_of_staplers, chest_drains, ondansetron]
features = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(predicted_class)

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[class_index] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Outcome=1. "
            f"The model predicts that your probability of Outcome=1 is {probability:.1f}%. "
            "This is only an estimate and should be interpreted together with clinical information. "
            "Please consult your healthcare professional for an appropriate evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Outcome=1. "
            f"The model predicts that your probability of Outcome=0 is {probability:.1f}%. "
            "This is only an estimate and should be interpreted together with clinical information. "
            "Please continue routine follow-up and seek medical advice if you have concerns."
        )

    st.write(advice)

    # MLPClassifier is not a tree model, so use KernelExplainer for SHAP values.
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(features, nsamples=100)
    shap_values_array = np.asarray(shap_values)

    # SHAP 0.46 returns (rows, features, classes); older releases may return a list.
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[class_index])[0]
    elif shap_values_array.ndim == 3:
        shap_row = shap_values_array[0, :, class_index]
    elif shap_values_array.ndim == 2:
        shap_row = shap_values_array[0]
    else:
        shap_row = shap_values_array.reshape(-1)

    expected_value = np.asarray(explainer.expected_value)
    base_value = (
        float(expected_value[class_index])
        if expected_value.ndim > 0
        else float(expected_value)
    )

    shap.force_plot(
        base_value,
        shap_row,
        features,
        feature_names=[display_feature_names.get(name, name) for name in feature_names],
        matplotlib=True,
        show=False,
        text_rotation=25,
        contribution_threshold=0,
    )
    shap_plot_path = APP_DIR / "shap_force_plot.png"
    plt.savefig(shap_plot_path, bbox_inches="tight", dpi=1200)
    st.image(str(shap_plot_path))
    plt.close()
