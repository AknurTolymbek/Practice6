from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer API")

model = joblib.load("model.joblib")


class CancerFeatures(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: CancerFeatures):
    data = np.array([[
        features.mean_radius, features.mean_texture,
        features.mean_perimeter, features.mean_area,
        features.mean_smoothness, features.mean_compactness,
        features.mean_concavity, features.mean_concave_points,
        features.mean_symmetry, features.mean_fractal_dimension,
        features.radius_error, features.texture_error,
        features.perimeter_error, features.area_error,
        features.smoothness_error, features.compactness_error,
        features.concavity_error, features.concave_points_error,
        features.symmetry_error, features.fractal_dimension_error,
        features.worst_radius, features.worst_texture,
        features.worst_perimeter, features.worst_area,
        features.worst_smoothness, features.worst_compactness,
        features.worst_concavity, features.worst_concave_points,
        features.worst_symmetry, features.worst_fractal_dimension
    ]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    return {
        "prediction": "benign" if prediction == 1 else "malignant",
        "prediction_label": int(prediction),
        "confidence": round(float(max(probability)) * 100, 2)
    }