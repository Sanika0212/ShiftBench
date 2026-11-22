.PHONY: help install lint types test sql-check check fetch score analyse serve

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

install: ## create the environment
	uv sync --python 3.12 --all-extras

lint: ## ruff check and format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

types: ## mypy, strict
	uv run mypy

test: ## the test suite, excluding the ones that need exported models
	uv run pytest -q --cov=shiftbench --cov-report=term --cov-fail-under=80

sql-check: ## the BigQuery DDL parses
	uv run --with sqlfluff sqlfluff lint infra/bigquery/schema.sql

check: lint types test sql-check ## everything CI runs

fetch: ## download the five domains into data/
	uv run shiftbench fetch

score: ## score every domain with the default detectors, resumable
	uv run shiftbench score

analyse: ## the matrices, budget curves, drift statistics and leaderboard
	uv run shiftbench analyse

serve: ## the leaderboard site
	uv run shiftbench serve
