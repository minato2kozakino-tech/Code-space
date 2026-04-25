import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg['paths']['intent_model']
        self.vectorizer = TfidfVectorizer(max_features=cfg['intent_classifier']['tfidf_max_features'])
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, texts, labels):
        print("[+] Training Intent Classifier (chat vs code)...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        self.save()

    def predict(self, text):
        if not self.is_trained:
            return "chat" # Default
        X = self.vectorizer.transform([text.lower()])
        return self.model.predict(X)[0]

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.model = data['model']
                self.is_trained = True
            return True
        return False
