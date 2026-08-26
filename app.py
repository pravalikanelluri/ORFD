import os
import re
import nltk # type: ignore
import pickle
import numpy as np # type: ignore
import spacy # type: ignore
from flask import Flask, render_template, request # type: ignore
from scipy.sparse import hstack # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load models and vectorizers
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'countvectorizer.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}.")
except Exception as e:
    print(f"Error loading {MODEL_PATH}: {e}")
    model = None

try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"CountVectorizer loaded successfully from {VECTORIZER_PATH}.")
except Exception as e:
    print(f"Error loading {VECTORIZER_PATH}: {e}.")
    print("Please make sure you have countvectorizer.pkl generated from eda_modelling.py")
    vectorizer = None

# Ensure NLTK datasets are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        pass

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Ensure spacy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    from spacy.cli import download # type: ignore
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords # type: ignore
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [str(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def normalize_text(text):
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)
    return normalized_text

recent_analyses = []

@app.route('/')
def home():
    return render_template('index.html', probability=0, result=None, recent_results=recent_analyses)

@app.route('/reports')
def reports():
    return render_template('reports.html', recent_results=recent_analyses)

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        job_description = request.form['description']
        
        if model is None or vectorizer is None:
            return render_template('result.html', 
                                   prediction_text='Error: Model or Vectorizer is not loaded.', 
                                   original_text=job_description, 
                                   probability=0, 
                                   result='ERROR')
        
        # Ensure type checker knows model and vectorizer are not None
        assert vectorizer is not None
        assert model is not None

        # 1. Clean Text
        cleaned_str = clean_text(job_description)
        
        # 2. Normalize Text
        normalized_str = normalize_text(cleaned_str)
        
        # 3. Extract POS Features on normalized string
        pos_tags = nltk.pos_tag(word_tokenize(normalized_str))
        pos_str = ' '.join(tag[1] for tag in pos_tags)
        
        # 4. Vectorize text and POS features
        text_matrix = vectorizer.transform([normalized_str])
        pos_matrix = vectorizer.transform([pos_str])
        
        # 5. Combine using hstack (as done in eda_modelling.py)
        combined_matrix = hstack([text_matrix, pos_matrix])
        
        # 6. Predict
        prediction = model.predict(combined_matrix)
        
        # Note: Decision trees and SVCs from scikit-learn might not always have predict_proba enabled by default
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(combined_matrix)[0][1]
        else:
            prob = 1.0 if prediction[0] == 1 else 0.0
            
        result = "FRAUDULENT" if prediction[0] == 1 else "GENUINE"
        
        # Extract a short name for display
        display_name = (job_description[:27] + '...') if len(job_description) > 30 else (job_description if job_description else "Empty Description")
        
        recent_analyses.insert(0, {"name": display_name, "tag": result})
        
        # Keep only the top 5 recent results
        if len(recent_analyses) > 5:
            recent_analyses.pop()
            
        return render_template('result.html', 
                               prediction_text=f'This job posting is likely: {result}', 
                               original_text=job_description, 
                               probability=prob*100, 
                               result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
