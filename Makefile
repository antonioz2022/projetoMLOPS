PYTHON = venv/bin/python
DVC    = $(PYTHON) -m dvc
PIP    = $(PYTHON) -m pip

.PHONY: help setup pipeline load preprocess train metrics mlflow clean

help:
	@echo "MLOps - Fraude"
	@echo "make pipeline  - Roda tudo"
	@echo "make mlflow    - Abre MLflow UI"
	@echo "make clean     - Limpa processados"

pipeline:
	$(DVC) repro

mlflow:
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -f data/processed/* models/* metrics/*