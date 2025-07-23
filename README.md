# Project LangChain: E2E PDF and Webpage Analysis Tool

An end-to-end tool for analyzing either webpages or PDF documents using **LangChain** and **Ollama** LLMs.  
Easily extract insights from web pages or local PDF files via a simple CLI.

---

## Features

- Analyze any URL directly from the web  
- Process local PDF files for quick insights  
- Simple command-line interface
- Can also run from Jupyter Notebook

---

## Installation

First, make sure you’ve installed your Python environment and dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Analyse a webpage:
```bash
python cli_analyzer.py --url https://www.pgatour.com/
```
Analyse a PDF File
```bash
python cli_analyzer.py --file "data/input/OptimalProteinIntakeandFrequncy.pdf"

```
