setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

run: setup
	venv/bin/pip train.py
	venv/bin/pip test.py

test: setup
	venv/bin/pip test.py