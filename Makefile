# Convenience targets for the DRC pipeline. Each wraps a module CLI so you can
# run a stage without remembering its flags. Override CONFIG to point elsewhere.
CONFIG ?= configs/base.yaml
PYTHON ?= python

.PHONY: help install test lint data tokenizer train eval analyze figures pipeline clean

help:
	@echo "Targets:"
	@echo "  install    pip install -e .[dev]"
	@echo "  test       run the pytest suite"
	@echo "  lint       ruff check the package"
	@echo "  data       download, parse, audit, build dose corpora"
	@echo "  tokenizer  train the shared BPE tokenizer"
	@echo "  train      run the full sweep (pilot-gated)"
	@echo "  eval       SLOR evaluation + n-gram baseline"
	@echo "  analyze    Hill fits, model comparison, clustering, decision"
	@echo "  figures    render the five paper figures"
	@echo "  pipeline   data -> tokenizer -> train -> eval -> analyze -> figures"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests

data:
	$(PYTHON) -m drc.data.download     --config $(CONFIG)
	$(PYTHON) -m drc.data.parse        --config $(CONFIG)
	$(PYTHON) -m drc.data.qa_audit     --config $(CONFIG) --judge manual
	$(PYTHON) -m drc.data.dose_corpora --config $(CONFIG)

tokenizer:
	$(PYTHON) -m drc.tokenizer.train_tokenizer --config $(CONFIG)

train:
	$(PYTHON) -m drc.train.sweep --config $(CONFIG)

eval:
	$(PYTHON) -m drc.eval.run_eval       --config $(CONFIG)
	$(PYTHON) -m drc.eval.ngram_baseline --config $(CONFIG)

analyze:
	$(PYTHON) -m drc.analysis.hill             --config $(CONFIG)
	$(PYTHON) -m drc.analysis.model_comparison --config $(CONFIG)
	$(PYTHON) -m drc.analysis.clustering       --config $(CONFIG)
	$(PYTHON) -m drc.analysis.decision         --config $(CONFIG)

figures:
	$(PYTHON) -m drc.analysis.figures --config $(CONFIG)

pipeline: data tokenizer train eval analyze figures

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
