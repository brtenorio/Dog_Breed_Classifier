# Makefile for the Dog_Breed_Classifier project

# Variables
PROJECT_NAME = Dog_Breed_Classifier
PYTHON = python3
PIP = pip
POETRY = poetry

# Targets
.PHONY: all setup install run clean test

all: setup install

setup:
	@echo "Setting up virtual environment..."
	$(POETRY) env use $(PYTHON)

install:
	@echo "Installing dependencies..."
	$(POETRY) install

run:
	@echo "Running the application..."
	$(POETRY) run streamlit run application/app.py --server.port 8080

test:
	@echo "Running tests..."
	$(POETRY) run pytest -v tests/test.py

retrain-model:
	@echo "Retrain and regenerate model file"
	$(POETRY) run python3 model_training/main.py

docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-run:
	@echo "Running application in Docker..."
	docker-compose up app

# to fix: test with docker fails due to data set directory not found
#docker-test:
#	@echo "Running tests in Docker..."
#	docker-compose run test

clean:
	@echo "Cleaning up..."
	$(POETRY) env remove $(PYTHON)

clear-cache:
	@echo "Cleaning up..."
	$(POETRY) cache clear --all .

docker-down:
	@echo "Running application in Docker..."
	docker-compose down

# Additional target for development purposes
dev-install:
	@echo "Installing development dependencies..."
	$(POETRY) install --with dev
