import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re
import joblib
from pdfminer.high_level import extract_text
import io


nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv('UpdatedResumeDataSet.csv')

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
print(df['Category'].unique())
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Resume'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'resume_scanner_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('resume_scanner_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['resume']
        # Check type of file
        if uploaded_file.filename.endswith('.pdf'):
            pdf_content = uploaded_file.read()
            text = extract_text(io.BytesIO(pdf_content))
        else:
            text = uploaded_file.read().decode('utf-8')
        
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        job_role = label_encoder.inverse_transform(prediction)[0]
        return f"This resume is best suited for: {job_role}"
    
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
