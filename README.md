# Calopic ML Trainer

A minimal, open-source command-line trainer for classical machine learning models, released by [Calopic](https://Calopic.top).

Website: [https://Calopic.top](https://Calopic.top)
X: [@CalopicAI](https://x.com/CalopicAI)

## What it is

`calopic-mltrainer` is a small CLI tool that lets you train and evaluate
classical ML classifiers on tabular CSV data with one command. v0.1 supports:

- Logistic Regression
- Random Forest

The goal is a clean, readable baseline trainer: no notebooks, no boilerplate,
no hidden magic. Point it at a CSV, pick a model, get accuracy and a saved model file.

## Install

```bash
pip install -r requirements.txt

Requires Python 3.9+.

## Usage
python cli.py fit --model logreg --data iris.csv --target target
python cli.py fit --model rf --data iris.csv --target target --save model.joblib

## Options:
Flag Description Default
--model logreg or rf required
--data path to CSV required
--target target column name required
--test-size test split ratio 0.2
--seed random seed 42
--save path to save trained model none

## Example

Generate the bundled Iris CSV:

python make_data.py

Then train:

python cli.py fit --model rf --data iris.csv --target target

Output：
metrics: {'test_acc': 0.9667}

## Roadmap
• v0.1: Logistic Regression, Random Forest (current)
• v0.2: XGBoost
• v0.3: CNN and Transformer for image data
• v0.4: YAML config files, cross-validation, feature importance export






