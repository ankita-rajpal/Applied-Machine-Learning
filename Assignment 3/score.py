import re
import string
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator
from typing import Tuple

def preprocess_text(text: str) -> str:
    """Preprocesses input text by tokenizing, removing stopwords and punctuation, and lemmatizing."""
    regex = r"\w+"
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    word_tokens = re.findall(regex, text)
    
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokens 
                       if word.lower() not in stop_words and word not in string.punctuation]
    
    return " ".join(filtered_tokens)

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """Scores a trained model on a text input.
    
    Args:
        text (str): The input text to score.
        model (sklearn.estimator): The trained model.
        threshold (float): The threshold for classification.
        
    Returns:
        Tuple[bool, float]: A tuple containing the prediction (bool) and propensity score (float).
    """
    preprocessed_text = preprocess_text(text)
    
    # Load vectorizer
    vectorizer = joblib.load("models/vectorizer.pkl")
    transformed_text = vectorizer.transform([preprocessed_text])
    
    # Get model prediction probability
    propensity = model.predict_proba(transformed_text)[:, 1][0]  # Assuming binary classification
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity

# Example Usage (assuming model is loaded elsewhere)
# model = joblib.load("models/model.pkl")
# prediction, propensity = score("that will be fine. Love you. Be safe", model, 0.5)
# print(prediction, propensity)
