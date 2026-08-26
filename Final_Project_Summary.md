# Final Project Summary: AI Fake Job Detection System

## 1. Project Objective
The primary objective of this project is to build an intelligent web application capable of classifying job postings as either **Real** (Authentic) or **Fake** (Fraudulent). With the rise of online job scams, this Artificial Intelligence-powered tool helps job seekers verify the authenticity of job descriptions before applying or sharing sensitive personal information.

## 2. Dataset Used
The model is trained on the **Real / Fake Job Posting Prediction** dataset (`fake_job_postings.csv`).
- **Total Records:** 17,880 job postings.
- **Key Features Available:** `title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, and `fraudulent` (target label).
- **Target Feature:** The model specifically focuses on analyzing the `description` column to detect linguistic patterns typically associated with scams. The `fraudulent` column acts as the boolean label (1 for Fake, 0 for Real).

## 3. Train-Test Split
The dataset was divided into training and testing sets to evaluate the model's performance on unseen data.
- **Split Ratio:** `test_size=0.25` (75% of compiling data for training, 25% for testing/validation).
- **Random State:** `32` for reproducible results.

## 4. How the Model Works and Is Trained
We utilized a **Natural Language Processing (NLP)** pipeline coupled with a **Deep Learning Recurrent Neural Network (RNN)**.

### Text Preprocessing:
1. **Cleaning:** All numbers and special characters are removed, retaining only alphabetic characters.
2. **Lowercasing:** Text is converted to lowercase for consistency.
3. **Tokenization & Stopwords Removal:** Sentences are split into individual words; common English stopwords are removed using the `nltk` library.
4. **Stemming:** Words are reduced to their root form using the **PorterStemmer** (e.g., "running" becomes "run").
5. **One-Hot Encoding:** The cleaned text is converted into one-hot representations bounded by a vocabulary size (`voc_size = 5000`).
6. **Padding:** Sequences are padded to ensure uniform input length (`maxlen = 40`, `padding='pre'`).

### Model Architecture (Bi-Directional LSTM):
The model is built using **TensorFlow/Keras** and consists of the following architecture:
1. **Embedding Layer:** Converts the one-hot encoded vectors into dense representations (`embedding_vector_features = 50`).
2. **Bidirectional LSTM Layer:** A powerful `Bidirectional(LSTM(100))` layer reads the sequence data in both directions (forward and backward) to understand the context of words better than a standard LSTM.
3. **Dropout Layer:** A `Dropout(0.3)` layer randomly turns off 30% of neurons to prevent overfitting to the training data.
4. **Dense Output Layer:** A final `Dense(1)` layer with a `sigmoid` activation function outputs a probability score between 0 and 1.

### Training Configurations:
- **Optimizer:** `adam`
- **Loss Function:** `binary_crossentropy`
- **Epochs:** `12`
- **Batch Size:** `64`

## 5. Structure of the Project
The project has a clean and modular structure:

```text
├── app.py                                         # The Flask server and backend logic
├── fake_job_model.h5                              # The pre-trained Keras model weight file
├── fake_job_postings.csv                          # The original dataset
├── real-or-fake-job-postings-with-bi-...ipynb     # Jupyter Notebook containing EDA and Model Training
├── requirements.txt                               # Python package dependencies
└── templates/
    └── index.html                                 # The frontend HTML/CSS/JS user interface
```

## 6. User Interface (UI)
The web interface acts as the presentation layer for the AI model. It has been highly customized for a premium user experience:
- **Aesthetic:** A modern "Glassmorphism" design with a frosted glass card layered over an animated, multi-colored glowing background layout.
- **Interactivity:** Fluid micro-animations, loading spinners, and immersive 3D parallax hover effects that react to mouse movements.
- **Functionality:** Users can seamlessly paste a job description. The application provides immediate visual cues indicating "Analyzing Neural Patterns" followed by a distinctly colored result card (🚨 **Red for Fraudulent**, ✅ **Green for Authentic**).

## 7. How to Run the Project
Follow these simple steps to run the application locally:

1. **Install Dependencies:**
   Ensure you have Python installed. Open your terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask Server:**
   Run the main server file using Python:
   ```bash
   python app.py
   ```

3. **Access the Web App:**
   Once the server starts, open your preferred web browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
   Paste a job description into the text area and click **Execute Deep Analysis** to test the system!
