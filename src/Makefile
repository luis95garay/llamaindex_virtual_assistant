.PHONY: start
start:
	uvicorn api.main:app --reload --port 9000

.PHONY: format
format:
	black .
	isort .