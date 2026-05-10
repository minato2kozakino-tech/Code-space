import csv
import numpy as np
import os
import re
import sys
import time
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

class MarkovChain:
    def __init__(self, order=2, smoothing="add_one"):
        self.order = order
        self.smoothing = smoothing
        self.lookup_table = defaultdict(Counter)
        self.starts = Counter()

    def fit(self, sentences):
        for sentence in sentences:
            words = sentence.split()
            if not words: continue
            self.starts[words[0]] += 1
            for i in range(len(words) - self.order):
                ctx = tuple(words[i:i+self.order])
                nxt = words[i+self.order]
                self.lookup_table[ctx][nxt] += 1
        return self

    def generate_next_word(self, context):
        if context not in self.lookup_table:
            return np.random.choice(list(self.starts.keys())) if self.starts else "..."
        counts = self.lookup_table[context]
        total = sum(counts.values())
        if self.smoothing == "add_one":
            v_size = 10000 
            probs = {w: (c + 1) / (total + v_size) for w, c in counts.items()}
        else:
            probs = {word: count / total for word, count in counts.items()}
        w_list, p_list = zip(*probs.items())
        p_list = np.array(p_list); p_list /= p_list.sum()
        return np.random.choice(w_list, p=p_list)


def load_conversational_csv(csv_path):
    cleaned = []
    metadata_pattern = re.compile(r'^(.*),Bloom-[^,]+,\d+,\d+,\d+$')

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() == 'human,bot':
                continue

            metadata_match = metadata_pattern.match(line)
            if metadata_match:
                human = metadata_match.group(1).strip()
                if human.startswith('"') and human.endswith('"'):
                    human = human[1:-1].replace('""', '"')
                cleaned.append([human, human])
                continue

            try:
                row = next(csv.reader([line], skipinitialspace=True))
            except Exception:
                row = [line]

            if len(row) >= 2:
                second = str(row[1]).strip()
                if re.fullmatch(r'[A-Za-z0-9_-]+', second) and all(str(c).strip().isdigit() for c in row[2:]):
                    cleaned.append([str(row[0]).strip(), str(row[0]).strip()])
                else:
                    cleaned.append([str(row[0]).strip(), second])
            else:
                cleaned.append([str(row[0]).strip(), str(row[0]).strip()])

    if not cleaned:
        raise ValueError(f"No usable conversation rows found in {csv_path}")
    return pd.DataFrame(cleaned, columns=['human', 'bot'])


def train_mars(brain, cfg):
    data_file = cfg['paths']['conversational_data']
    epochs = cfg['trainer']['epochs']
    lr = cfg['trainer']['learning_rate']
    n_gram = cfg['trainer']['n_gram']
    grad_clip = cfg['trainer']['gradient_clip_norm']

    if not os.path.exists(data_file):
        print("[-] Conversational data file not found.")
        return
    
    print("[+] Loading and preparing conversational data...")
    try:
        df = load_conversational_csv(data_file)
    except Exception as e:
        print(f"[-] Error parsing CSV: {e}")
        return
    if 'bot' not in df.columns:
        print("[-] CSV must have 'bot' column.")
        return
    lines = df['bot'].dropna().str.lower().tolist()

    sentences_ml, labels_ml, tokens_nn = [], [], []
    for l in lines:
        sentences_ml.append(l)
        if "function" in l or "local" in l or "end" in l: labels_ml.append("lua")
        elif any(ord(c) > 127 for c in l): labels_ml.append("vn")
        else: labels_ml.append("en")
        tokens_nn.extend(l.split())

    print("[+] Fitting specialized components (TF-IDF, Classifier, Markov)...")
    brain.vectorizer = TfidfVectorizer(max_features=5000)
    x_tfidf = brain.vectorizer.fit_transform(sentences_ml)
    brain.lang_classifier = MultinomialNB().fit(x_tfidf, labels_ml)
    brain.markov_model = MarkovChain(order=2).fit(sentences_ml)  # Default order
    
    brain.expand(list(set(tokens_nn)))
    
    brain.save()
    print("[!] Specialized components fitted and model saved.")
