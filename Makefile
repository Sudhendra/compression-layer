.PHONY: install dev test lint typecheck format clean generate validate train-local train-cloud eval all

# Setup
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .

dev:
	pip install -e ".[dev]"
	pip install -U mlx-lm tinker

# Quality
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ --fix

typecheck:
	mypy src/

format:
	ruff format src/ scripts/

check: lint typecheck test

# Pipeline
generate:
	python scripts/generate_seed.py \
		--config configs/generation.yaml \
		--count 1000

validate:
	python scripts/validate_batch.py \
		--config configs/generation.yaml

# Training
train-local:
	python -m mlx_lm.lora \
		--model mlx-community/Qwen3-4B-Instruct-4bit \
		--train \
		--data ./data/validated \
		--iters 500

train-cloud:
	python scripts/train_tinker.py \
		--config configs/training.yaml \
		--output models/adapters/tinker

eval:
	python scripts/evaluate.py \
		--model models/checkpoints/adapter \
		--test data/validated/test.jsonl

# Full pipeline
all: generate validate train-cloud eval

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-data:
	rm -rf data/seed/* data/validated/* data/cache/*

clean-models:
	rm -rf models/checkpoints/* adapters/*
