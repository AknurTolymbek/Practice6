# Breast Cancer ML API

ML model deployed with FastAPI and Docker.

## Project Structure
- `train.py` — trains and saves the model
- `main.py` — FastAPI application
- `model.joblib` — saved model
- `requirements.txt` — dependencies
- `Dockerfile` — container instructions

## How to Run

### Locally
pip install -r requirements.txt
uvicorn main:app --reload

### With Docker
docker build -t ml-api .
docker run -p 8000:8000 ml-api

## Endpoints
- GET / — health check
- POST /predict — returns prediction (malignant/benign)

## Model
- Dataset: Breast Cancer Wisconsin
- Algorithm: Random Forest Classifier
- Accuracy: ~96%