.PHONY: install train serve test lint docker-build docker-run clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run data processing and model training
train:
	python -m src.data_processing
	python -m src.train

# Run the API server locally
serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	pytest tests/ -v --tb=short

# Lint code
lint:
	ruff check src/ tests/

# Build Docker image
docker-build:
	docker build -t fraud-detection-api .

# Run Docker container
docker-run:
	docker run -p 8000:8000 -v ./models:/app/models fraud-detection-api

# Run full stack with docker-compose
docker-up:
	docker compose up --build

# Stop docker-compose
docker-down:
	docker compose down

# Start MLflow UI
mlflow-ui:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

# Clean artifacts
clean:
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf mlruns/
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
