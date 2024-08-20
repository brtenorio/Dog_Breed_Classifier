# Makefile for the Dog_Breed_Classifier project

# Variables
PROJECT_NAME = Dog_Breed_Classifier
PYTHON = python3
PIP = pip
POETRY = poetry

# Targets
.PHONY: all setup install run clean test

all: setup install test

setup:
	@echo "Setting up virtual environment..."
	$(POETRY) env use $(PYTHON)

install:
	@echo "Installing dependencies..."
	$(POETRY) install

run-app:
	@echo "Running the application..."
	$(POETRY) run streamlit run application/app.py --server.port 8080

run-api:
	@echo "Running the API..."
	$(POETRY) run $(PYTHON) api/api.py

test:
	@echo "Running tests..."
	$(POETRY) run pytest -v tests/test.py

retrain-model:
	@echo "Retrain and regenerate model file"
	$(POETRY) run $(PYTHON) dog_breed_classifier/main.py

docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-run-app:
	@echo "Running application in Docker..."
	docker-compose up app

docker-run-api:
	@echo "Running application in Docker..."
	docker-compose up api

docker-test:
	@echo "Running tests in Docker..."
	docker-compose run test

docker-down:
	@echo "Running application in Docker..."
	docker-compose down

clean:
	@echo "Cleaning up..."
	$(POETRY) env remove $(PYTHON)

clear-cache:
	@echo "Cleaning up..."
	$(POETRY) cache clear --all .

