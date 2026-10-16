.PHONY: run replay validate

IMAGE ?= hackapertus-uzh-rsd

run:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE)

replay:
	python src/submission_replay.py data/results/final_test_8b_summary.json data/results/final_test_70b_summary.json

validate:
	python -m compileall -q src tests
	PYTHONPATH=src python -m pytest -q
