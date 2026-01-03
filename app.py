import streamlit as st
import joblib
import fitz
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw]
    return " ".join(words)

def predict_category(text):
    clean = preprocess(text)
    vector = tfidf.transform([clean]).toarray()
    prediction = svm_model.predict(vector)
    return prediction[0]

st.title("📄 Resume Screening App (SVM)")
st.write("Upload your resume (PDF or TXT) to predict category.")

file = st.file_uploader("Choose file", type=["pdf", "txt"])

if file:
    if file.type == "application/pdf":
        resume_text = extract_text_from_pdf(file)
    else:
        resume_text = file.read().decode("utf-8")

    st.subheader("Preview")
    st.write(resume_text[:800])

    if st.button("Predict"):
        result = predict_category(resume_text)
        st.success(f"Predicted Category: {result}")
