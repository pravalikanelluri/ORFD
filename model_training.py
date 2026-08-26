import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import hashing_trick
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(f"TensorFlow Version: {tf.__version__}")

# Load Dataset
print("Loading dataset...")
df1 = pd.read_csv('fake_job_postings.csv', engine='python')
df = df1[['description', 'requirements', 'fraudulent']]

# Drop Nan Values
df = df.dropna()
print(f"Dataset shape after dropna: {df.shape}")

# Get Features and Labels
X = df.drop('fraudulent', axis=1)
y = df['fraudulent']

message = X.copy()
message.reset_index(inplace=True)

# Preprocessing
print("Downloading stopwords and preprocessing text...")
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []

for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['description'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Onehot Representation
print("Encoding text...")
voc_size = 5000
onehot_repr = [hashing_trick(words, voc_size, hash_function='md5') for words in corpus]

# Embedding / Padding
sent_length = 40
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Create Model
print("Building model...")
embedding_vector_features = 50
model1 = Sequential()
model1.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

# Prepare Training Data
X_final = np.array(embedded_docs)
y_final = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=32)

# Train the Model
print("Training model...")
model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=64)

# Model Performance and Accuracy
print("Evaluating model...")
y_pred = (model1.predict(X_test) > 0.5).astype("int32")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred)}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
model1.save('fake_job_model.h5')
print('Model saved successfully as fake_job_model.h5')

import pickle
# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully!")