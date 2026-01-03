import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---------- Preprocess Function ----------
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Remove numbers & symbols
    text = text.lower()                          # Lowercase
    tokens = word_tokenize(text)                 # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# ---------- Load Dataset ----------
df = pd.read_csv('dataset.csv')

# Clean text column
df['cleaned_resume'] = df['Resume'].apply(preprocess_text)

# ---------- Feature Extraction ----------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_resume']).toarray()
y = df['Category']

# ---------- Train / Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Model ----------
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# ---------- Predictions ----------
y_pred = svm_model.predict(X_test)

# ---------- Accuracy ----------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ---------- Classification Report ----------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------- Cross Validation ----------
scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())

# ---------- Misclassified Examples ----------
misclassified = []
for text, real, pred in zip(df['Resume'], y_test, y_pred):
    if real != pred:
        misclassified.append((real, pred, text[:200]))

print("\nExamples of misclassified resumes:")
for i in misclassified[:5]:
    print("\nActual:", i[0])
    print("Predicted:", i[1])
    print("Snippet:", i[2])

# ---------- Save Model + Vectorizer ----------
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel + Vectorizer Saved!")
