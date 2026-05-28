# make commands for common tasks

.PHONY: help install train test serve docker-build docker-run clean

help:
	@echo "available commands:"
	@echo "  make install      - install python packages"
	@echo "  make train        - train the model"
	@echo "  make test         - run unit tests"
	@echo "  make serve        - start api locally"
	@echo "  make docker-build - build docker image"
	@echo "  make docker-run   - run with docker-compose"
	@echo "  make monitor      - check model performance"
	@echo "  make clean        - remove temporary files"

install:
	pip install -r requirements.txt

train:
	python run_pipeline.py

test:
	pytest tests/ -v

serve:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t energy-forecast-api .

docker-run:
	docker-compose up --build

docker-stop:
	docker-compose down

monitor:
	python src/monitor.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
