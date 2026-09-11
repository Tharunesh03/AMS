PY ?= .venv/bin/python
U  ?= 1

.PHONY: help venv install prepare smoke train reference evaluate demo serve test docker clean

help:            ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

venv install:    ## create .venv and install requirements
	$(PY) -m venv .venv || true
	.venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

smoke:           ## tiny end-to-end run (seconds) to validate the pipeline
	$(PY) train.py --config configs/smoke.yaml

train:           ## full dataset inspection + candidate comparison + production model
	$(PY) train.py --config configs/default.yaml

reference:       ## rebuild models/reference_stats.json (evidence engine reference)
	$(PY) -m vfa reference

evaluate:        ## re-measure the production artifact on the held-out test split
	$(PY) -m vfa evaluate

demo:            ## predict the bundled samples from the terminal
	$(PY) -m vfa demo

serve:           ## run the FastAPI app on :8000 (loads models/, never retrains)
	$(PY) -m uvicorn app.server:app --host 0.0.0.0 --port 8000

test:            ## unit + integration tests
	$(PY) -m pytest -q

docker:          ## build the CPU-only deployment image
	docker build -t vfa-ai-detector .

clean:
	rm -rf artifacts reports/figures models/ai_detector __pycache__ .pytest_cache
