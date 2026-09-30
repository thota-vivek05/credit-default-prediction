# Credit Default Prediction

A machine learning project for predicting the probability of loan default.
The project exposes the model through a FastAPI service and stores prediction
records in PostgreSQL.

I built this repository as a hands-on project to understand credit-risk
prediction, feature engineering, model serving, explainability, and API
development.

## Features

- Predict default risk for one applicant with `POST /predict`.
- Score multiple applicants with `POST /batch`.
- Explain a stored prediction with `GET /explain/{prediction_id}`.
- Check the API and model status with `GET /health`.
- Generate SHAP-based explanations for predictions.
- Store prediction results in PostgreSQL.

## Technologies

- Python 3.11
- FastAPI and Uvicorn
- XGBoost and scikit-learn
- SHAP
- SQLAlchemy and PostgreSQL
- Docker and pytest

## Project structure

```text
app/
  api/       API route handlers
  db/        Database connection and models
  ml/        Model, feature pipeline, training, and explanations
  schemas/   Request and response schemas
tests/       API tests
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/thota-vivek05/credit-default-prediction.git
cd credit-default-prediction
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
pip install -r requirements.txt
```

The API expects a trained model at:

```text
models/xgboost_model.pkl
```

After the model and database are configured, start the API with:

```bash
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. The interactive documentation is
available at `http://localhost:8000/docs`.

## Docker

```bash
docker compose up --build
```

The Docker configuration starts the API and PostgreSQL services. The model file
must be available in the local `models` directory.

## Example request

```json
{
  "age": 45,
  "income": 6500,
  "debt_ratio": 0.35,
  "monthly_income": 6500,
  "open_credit_lines": 4,
  "times_late_30_59": 0,
  "times_late_60_89": 0,
  "times_late_90": 0,
  "revolving_utilization": 0.28,
  "real_estate_loans": 1,
  "dependents": 2
}
```

Send the request to `POST /predict` as JSON. The response contains the default
probability, risk level, decision, confidence, and explanation summary.

## Configuration

Copy `.env.example` to `.env` and update the values for your environment:

```env
DATABASE_URL=postgresql://cs_user:cs_pass@localhost:5432/credit_scoring
MODEL_PATH=./models/xgboost_model.pkl
DEFAULT_THRESHOLD=0.15
LOW_RISK_THRESHOLD=0.10
MEDIUM_RISK_THRESHOLD=0.25
```

## Tests

```bash
pytest tests/ -v
```

## Acknowledgements

This project was developed for learning purposes with reference to existing
credit-scoring implementations.

The current repository is maintained under my own GitHub account, and future
changes and improvements will be documented here.
