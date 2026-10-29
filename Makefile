.PHONY: check predict train
PYTHON ?= .venv/bin/python

check:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) scripts/check.py

predict:
	OPENBLAS_NUM_THREADS=4 $(PYTHON) run.py

train:
	OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $(PYTHON) train.py --workers 6
