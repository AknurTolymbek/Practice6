import streamlit as st
import requests

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🔬")
st.title("🔬 Breast Cancer Classifier")
st.markdown("Enter tumor parameters to get a prediction")

with st.form("prediction_form"):
    st.subheader("Mean Values")
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.number_input("Mean Radius", value=14.0)
        mean_texture = st.number_input("Mean Texture", value=19.0)
        mean_perimeter = st.number_input("Mean Perimeter", value=91.0)
        mean_area = st.number_input("Mean Area", value=654.0)
        mean_smoothness = st.number_input("Mean Smoothness", value=0.10, format="%.4f")
        mean_compactness = st.number_input("Mean Compactness", value=0.10, format="%.4f")
        mean_concavity = st.number_input("Mean Concavity", value=0.09, format="%.4f")
        mean_concave_points = st.number_input("Mean Concave Points", value=0.05, format="%.4f")
        mean_symmetry = st.number_input("Mean Symmetry", value=0.18, format="%.4f")
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06, format="%.4f")

    with col2:
        radius_error = st.number_input("Radius Error", value=0.4, format="%.4f")
        texture_error = st.number_input("Texture Error", value=1.2, format="%.4f")
        perimeter_error = st.number_input("Perimeter Error", value=2.8, format="%.4f")
        area_error = st.number_input("Area Error", value=40.0)
        smoothness_error = st.number_input("Smoothness Error", value=0.007, format="%.4f")
        compactness_error = st.number_input("Compactness Error", value=0.025, format="%.4f")
        concavity_error = st.number_input("Concavity Error", value=0.032, format="%.4f")
        concave_points_error = st.number_input("Concave Points Error", value=0.012, format="%.4f")
        symmetry_error = st.number_input("Symmetry Error", value=0.02, format="%.4f")
        fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.004, format="%.4f")

    with col3:
        worst_radius = st.number_input("Worst Radius", value=16.0)
        worst_texture = st.number_input("Worst Texture", value=25.0)
        worst_perimeter = st.number_input("Worst Perimeter", value=107.0)
        worst_area = st.number_input("Worst Area", value=880.0)
        worst_smoothness = st.number_input("Worst Smoothness", value=0.13, format="%.4f")
        worst_compactness = st.number_input("Worst Compactness", value=0.25, format="%.4f")
        worst_concavity = st.number_input("Worst Concavity", value=0.27, format="%.4f")
        worst_concave_points = st.number_input("Worst Concave Points", value=0.11, format="%.4f")
        worst_symmetry = st.number_input("Worst Symmetry", value=0.29, format="%.4f")
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.08, format="%.4f")

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

if submitted:
    payload = {
        "mean_radius": mean_radius, "mean_texture": mean_texture,
        "mean_perimeter": mean_perimeter, "mean_area": mean_area,
        "mean_smoothness": mean_smoothness, "mean_compactness": mean_compactness,
        "mean_concavity": mean_concavity, "mean_concave_points": mean_concave_points,
        "mean_symmetry": mean_symmetry, "mean_fractal_dimension": mean_fractal_dimension,
        "radius_error": radius_error, "texture_error": texture_error,
        "perimeter_error": perimeter_error, "area_error": area_error,
        "smoothness_error": smoothness_error, "compactness_error": compactness_error,
        "concavity_error": concavity_error, "concave_points_error": concave_points_error,
        "symmetry_error": symmetry_error, "fractal_dimension_error": fractal_dimension_error,
        "worst_radius": worst_radius, "worst_texture": worst_texture,
        "worst_perimeter": worst_perimeter, "worst_area": worst_area,
        "worst_smoothness": worst_smoothness, "worst_compactness": worst_compactness,
        "worst_concavity": worst_concavity, "worst_concave_points": worst_concave_points,
        "worst_symmetry": worst_symmetry, "worst_fractal_dimension": worst_fractal_dimension,
    }

    try:
        response = requests.post("http://fastapi:8000/predict", json=payload)
        result = response.json()

        prediction = result["prediction"]
        confidence = result["confidence"]

        if prediction == "malignant":
            st.error("⚠️ Result: MALIGNANT tumor")
        else:
            st.success("✅ Result: BENIGN tumor")

        st.metric("Model Confidence", f"{confidence}%")

    except Exception as e:
        st.error(f"Connection error with API: {e}")