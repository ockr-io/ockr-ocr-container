# ockr-ocr-container

<p align="left">
<img alt="Release" src="https://github.com/ockr-io/ockr-ocr-container/actions/workflows/release.yaml/badge.svg?branch=main" />
<img alt="Publish" src="https://github.com/ockr-io/ockr-ocr-container/actions/workflows/publish.yaml/badge.svg?branch=main" />
<a href="https://conventionalcommits.org"><img alt="conventionalcommits" src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits" /></a>
</p>

The Ockr OCR container offers a FastAPI service that can be used to extract information from documents.
It is based on the Ockr model zoo. A list of available models can be found [here](https://github.com/ockr-io/ockr-model-zoo).

## Getting started

### Prerequisites

- Python ^3.11
- Poetry ^1.5.1

### Installation

```zsh
git clone https://github.com/ockr-io/ockr-ocr-container.git
cd ockr-ocr-container
poetry install
```
### Usage

```zsh
poetry run python app.py
```

### API documentation

The documentation will be available at: http://127.0.0.1:5001/docs
