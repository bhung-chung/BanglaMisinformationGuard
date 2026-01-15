.PHONY: download-data setup

setup:
	pip install -r requirements.txt

download-data:
	python src/download_data.py
