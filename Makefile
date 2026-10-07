PYTHON ?= python3
YEAR ?= 2024

.PHONY: install test build-data backtest

install:
	$(PYTHON) -m pip install -e '.[dev,train]'

test:
	$(PYTHON) -m pytest -q

build-data:
	$(PYTHON) -m energy.cli build-training-data --project-root . --year $(YEAR)

backtest:
	$(PYTHON) -m energy.cli run-dynamic-charge-economic-backtest \
		--project-root . --start 2025-01-01 --end 2025-09-30
