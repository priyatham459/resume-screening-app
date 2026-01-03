import streamlit as st
import joblib
import fitz
import re
import nltk
from nltk.corpus import stopwords
from skills_list import skills   # NEW

nltk.download('stopwords')

svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# ---------- Extract PDF ----------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ---------- Preprocess ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw]
    return " ".join(words)


# ---------- Predict Category ----------
def predict_category(text):
    clean = preprocess(text)
    vector = tfidf.transform([clean]).toarray()
    prediction = svm_model.predict(vector)
    return prediction[0]


# ---------- Extract Skills ----------
def extract_skills(text):
    text = text.lower()
    found = []

    for skill in skills:
        if skill in text:
            found.append(skill)

    return list(set(found))


# ---------- UI ----------
st.title("📄 Resume Screening App (SVM)")
st.write("Upload your resume (PDF or TXT) to predict category and extract skills.")

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
        skillset = extract_skills(resume_text)

        st.success(f"Predicted Category: {result}")

        st.subheader("🧠 Extracted Skills")
        st.write(", ".join(skillset) if skillset else "No major skills detected")
