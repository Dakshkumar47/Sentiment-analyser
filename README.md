# Sentiment Analyser (ML Algorithms)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GitHub stars](https://img.shields.io/github/stars/Dakshkumar47/Sentiment-analyser.svg?style=social&label=Star&maxAge=2592000)](https://github.com/Dakshkumar47/Sentiment-analyser/stargazers/)

## Project Overview
This project builds a sentiment classifier to detect positive, negative, or neutral polarity in text (e.g., reviews, tweets). It uses ML algorithms like Naive Bayes, Logistic Regression, and SVM, with NLTK for tokenization, stemming, and TF-IDF vectorization. Trained on public datasets, it achieves ~85% accuracy.

This NLP tool showcases my skills in text classification and is part of my B.Tech AI & ML projects.

## Features
- **Preprocessing**: Tokenization, stopword removal, stemming/lemmatization via NLTK.
- **Vectorization**: TF-IDF for feature extraction.
- **Models**: Multiple classifiers tuned with GridSearchCV.
- **Prediction**: Classifies new text with confidence scores.

## Repository Structure
- `trainer.py`: Script for training and saving models.
- `predictor.py`: For inference on new text.
- `dataset.csv`: Sample sentiment-labeled data.
- `README.md`: This file.

## Requirements
- Python 3.8+
- Libraries: `nltk`, `scikit-learn`, `pandas`

## Install via:
bash
pip install nltk scikit-learn pandas


**How to Run**
## Train:
bashpython trainer.py


## Predict:
python# Example in predictor.py
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model...
text = "This movie was amazing!"
pred = model.predict([text])
print('Sentiment:', pred[0])  # e.g., 'positive'
