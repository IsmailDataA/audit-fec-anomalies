.PHONY: install run test
install:
	pip install -r requirements.txt
run:
	python run_audit.py
test:
	pytest -q
