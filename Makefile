.PHONY: install demo test lint all

install:
	pip install -r requirements.txt

demo:
	PYTHONPATH=. python examples/walkthrough.py

test:
	python -m unittest discover -s tests -t . -v

lint:
	ruff check .

all: lint test demo
