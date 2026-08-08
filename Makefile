.DEFAULT_GOAL := help
.PHONY: help install dev run seed reseed test lint fmt check clean smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Sync the virtualenv from uv.lock
	uv sync

dev: ## Run the API with autoreload on :8000
	uv run uvicorn app.main:app --reload --port 8000

run: ## Run the API without reload (production-shaped)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

seed: ## Seed demo data (no-op if complaints already exist)
	uv run python -m scripts.seed

reseed: ## Wipe and regenerate the demo dataset
	uv run python -m scripts.seed --reset

test: ## Run the test suite
	uv run pytest -q

lint: ## Lint and autofix
	uv run ruff check --fix .

fmt: ## Format
	uv run ruff format .

check: lint test ## Lint then test

smoke: ## Hit the key endpoints against a running server on :8000
	@curl -sf http://127.0.0.1:8000/health | head -c 400; echo
	@curl -sf -X POST http://127.0.0.1:8000/api/v1/complaints \
		-H 'Content-Type: application/json' \
		-d '{"description":"Large pothole on Main University Road near the school gate.","location_text":"Block 5, Gulshan-e-Iqbal, Karachi","consent":true}' | head -c 400; echo

clean: ## Remove caches and the local database
	rm -rf .pytest_cache .ruff_cache civic.db
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
