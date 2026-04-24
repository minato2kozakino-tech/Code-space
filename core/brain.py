import numpy as np
import os
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class MarsBrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg['paths']['gaton_model']
        self.hidden_size = cfg['brain']['hidden_size']
        self.w2i, self.i2w = {}, {}
        
        # Hỗ trợ cả 2 lớp và 3 lớp (Deep)
        self.W_in = None
        self.W_h1, self.b_h1 = None, None
        self.W_h2, self.b_h2 = None, None
        self.W_out, self.b_out = None, None
        
        self.vectorizer = None 
        self.lang_classifier = None 
        self.markov_model = None
        self.load_model()

    def load_model(self):
        path = self.model_path
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.W_in = data.get("W_in")
                    # Load theo style Deep Gaton trước
                    self.W_h1 = data.get("W_h1") or data.get("W_h")
                    self.b_h1 = data.get("b_h1") or data.get("b_h")
                    self.W_h2 = data.get("W_h2")
                    self.b_h2 = data.get("b_h2")
                    self.W_out = data.get("W_out")
                    self.b_out = data.get("b_out")
                    
                    self.w2i, self.i2w = data["w2i"], data["i2w"]
                    self.vectorizer = data.get("vectorizer")
                    self.lang_classifier = data.get("lang_classifier")
                    self.markov_model = data.get("markov_model")
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                self.init_empty()
        else: self.init_empty()

    def init_empty(self):
        self.W_in = np.empty((0, self.hidden_size), dtype=np.float32)
        self.W_h1 = np.empty((self.hidden_size, self.hidden_size), dtype=np.float32)
        self.b_h1 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.W_out = np.empty((self.hidden_size, 0), dtype=np.float32)
        self.b_out = np.empty((1, 0), dtype=np.float32)
        self.w2i, self.i2w = {}, {}

    def expand(self, words):
        new = [w for w in words if w not in self.w2i]
        if not new: return
        curr = len(self.w2i)
        for i, w in enumerate(new):
            self.w2i[w], self.i2w[curr + i] = curr + i, w
        std = self.cfg['brain']['init_std']
        e_in = (np.random.randn(len(new), self.hidden_size) * std).astype(np.float32)
        self.W_in = np.vstack([self.W_in, e_in]) if self.W_in.size else e_in
        e_out = (np.random.randn(self.W_h1.shape[0] if self.W_h1 is not None else self.hidden_size, len(new)) * std).astype(np.float32)
        self.W_out = np.hstack([self.W_out, e_out]) if self.W_out.size else e_out
        e_b_out = np.zeros((1, len(new)), dtype=np.float32)
        self.b_out = np.hstack([self.b_out, e_b_out]) if self.b_out.size else e_b_out

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "W_in": self.W_in, "W_h1": self.W_h1, "b_h1": self.b_h1,
                "W_h2": self.W_h2, "b_h2": self.b_h2,
                "W_out": self.W_out, "b_out": self.b_out,
                "w2i": self.w2i, "i2w": self.i2w, 
                "vectorizer": self.vectorizer,
                "lang_classifier": self.lang_classifier,
                "markov_model": self.markov_model
            }, f)
