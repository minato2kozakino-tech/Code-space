import numpy as np
import os
import sys
import time
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

def train_mars(brain, cfg):
    data_file = cfg['paths']['data_file']
    epochs = cfg['trainer']['epochs']
    lr = cfg['trainer']['learning_rate']
    n_gram = cfg['trainer']['n_gram']
    grad_clip = cfg['trainer']['gradient_clip_norm']

    if not os.path.exists(data_file):
        print("[-] Data file not found.")
        return
    
    print("[+] Loading and preparing data...")
    with open(data_file, "r", encoding="utf-8") as f:
        lines = [l.strip().lower() for l in f if l.strip()]

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
    brain.markov_model = MarkovChain(order=cfg['markov_chain']['order']).fit(sentences_ml)
    
    brain.expand(list(set(tokens_nn)))
    t_ids = np.array([brain.w2i[t] for t in tokens_nn if t in brain.w2i], dtype=np.uint32)
    n_samples = len(t_ids) - n_gram
    if n_samples <= 0: return

    print("[+] Starting Deep Training (Press Ctrl+C to save and stop)...")
    total_start = time.time()
    
    try:
        for ep in range(1, epochs + 1):
            ep_start = time.time()
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for i, idx in enumerate(indices):
                if i % 100 == 0 or i == n_samples - 1:
                    now = time.time()
                    elap_ep = now - ep_start
                    speed = (i + 1) / elap_ep if elap_ep > 0 else 0
                    eta_ep = (n_samples - (i + 1)) / speed if speed > 0 else 0
                    
                    elap_total = now - total_start
                    avg_ep_time = elap_total / ep
                    eta_total = (epochs - ep) * avg_ep_time + eta_ep
                    
                    pct = (i + 1) / n_samples
                    b_len = 15
                    fill = int(b_len * pct)
                    bar = "█" * fill + "░" * (b_len - fill)
                    
                    msg = "\rEpoch {:02d}/{} |{}| {:.1%} [EP ETA: {:.1f}s, Total ETA: {:.1f}s]".format(
                        ep, epochs, bar, pct, eta_ep, eta_total
                    )
                    print(msg, end='', flush=True)

                ctx = t_ids[idx : idx + n_gram]
                target = t_ids[idx + n_gram]
                
                h1 = np.mean(brain.W_in[ctx], axis=0).reshape(1, -1)
                z2 = np.dot(h1, brain.W_h) + brain.b_h
                h2 = relu(z2)
                logits = np.dot(h2, brain.W_out) + brain.b_out
                
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                
                d_logits = probs.copy()
                d_logits[0, target] -= 1
                
                d_W_out = np.dot(h2.T, d_logits)
                d_b_out = d_logits
                d_h2 = np.dot(d_logits, brain.W_out.T)
                d_z2 = d_h2 * relu_derivative(z2)
                d_W_h = np.dot(h1.T, d_z2)
                d_b_h = d_z2
                d_h1 = np.dot(d_z2, brain.W_h.T)
                
                for g in [d_W_out, d_b_out, d_W_h, d_b_h, d_h1]:
                    np.clip(g, -grad_clip, grad_clip, out=g)

                brain.W_out -= lr * d_W_out
                brain.b_out -= lr * d_b_out
                brain.W_h -= lr * d_W_h
                brain.b_h -= lr * d_b_h
                brain.W_in[ctx] -= lr * d_h1 / n_gram
                
            print(" | Epoch {} Done in {:.1f}s".format(ep, time.time() - ep_start))
            
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving current state...")

    brain.save()
    print("\n[!] Gaton Model saved successfully.")
