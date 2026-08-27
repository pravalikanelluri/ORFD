- > Online Recruitment Fraud Detection
  Online job platforms can contain fake and fraudulent job postings.This project uses Machine Learning and NLP to identify whether a job posting is real or fake, helping users avoid potential recruitment scams.

- > Features
- Detects potentially fake job postings
- Uses machine learning for classification
- Applies text preprocessing and feature extraction
- Provides a simple Flask-based web interface
- Includes model training and evaluation and LSTM for text classification.

- > Technologies Used
-Python
-Machine Learning
-Natural Language Processing (NLP)
-Flask
-Scikit-learn
-Pandas
-NumPy

- > Project Structure
ORFD/
│
├── templates/
├── app.py
├── model_training.py
├── fix_model.py
├── real_or_fake_job_postings_with_bi_directional_lstm.ipynb
├── fake_job_postings.csv
├── model.pkl
├── multinomial_nb.pkl
├── countvectorizer.pkl
├── verified_fraud_examples.md
├── style.css
└── .gitignore
