import numpy as np
import os
import pickle
import json
import sys
import time
from collections import Counter

# --- MARS-1.2 CONFIG ---
MODEL_DIR = "models"
MARS_MODEL = os.path.join(MODEL_DIR, "mars-1.origon")
DATA_FILE = "data_300kb.txt"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

class MarsBrain:
    def __init__(self, hidden_size=15):
        self.hidden_size = hidden_size
        self.w2i, self.i2w = {}, {}
        self.W1, self.W2 = None, None
        self.tfidf = {}
        self.load_model()

    def load_model(self):
        if os.path.exists(MARS_MODEL):
            try:
                with open(MARS_MODEL, "rb") as f:
                    data = pickle.load(f)
                    self.W1, self.W2 = data["W1"], data["W2"]
                    self.w2i, self.i2w = data["w2i"], data["i2w"]
                    self.tfidf = data.get("tfidf", {})
                    print("[+] Mars-1.2: Brain ready.")
            except: self.init_empty()
        else: self.init_empty()

    def init_empty(self):
        self.W1 = np.empty((0, self.hidden_size), dtype=np.float32)
        self.W2 = np.empty((self.hidden_size, 0), dtype=np.float32)
        self.w2i, self.i2w = {}, {}
        self.tfidf = {}

    def expand(self, words):
        new = [w for w in words if w not in self.w2i]
        if not new: return
        curr = len(self.w2i)
        for i, w in enumerate(new):
            self.w2i[w], self.i2w[curr + i] = curr + i, w
        e1 = (np.random.randn(len(new), self.hidden_size) * 0.1).astype(np.float32)
        self.W1 = np.vstack([self.W1, e1]) if self.W1.size else e1
        e2 = (np.random.randn(self.hidden_size, len(new)) * 0.1).astype(np.float32)
        self.W2 = np.hstack([self.W2, e2]) if self.W2.size else e2

    def save(self):
        with open(MARS_MODEL, "wb") as f:
            pickle.dump({"W1": self.W1, "W2": self.W2, "w2i": self.w2i, "i2w": self.i2w, "tfidf": self.tfidf}, f)

def train_mars(brain, epochs=20):
    if not os.path.exists(DATA_FILE): return
    print("[+] Mars-1.2: Training Mode...")
    with open(DATA_FILE, "r", encoding="utf-8") as f: tokens = f.read().lower().split()
    counts = Counter(tokens)
    total = len(tokens)
    brain.expand(list(set(tokens)))
    brain.tfidf = {w: np.log(total/(c+1)) for w, c in counts.items()}
    ids = np.array([brain.w2i[t] for t in tokens], dtype=np.uint32)
    n_samples, lr = len(ids) - 5, 0.05
    for ep in range(1, epochs + 1):
        start = time.time()
        idx_list = np.arange(n_samples); np.random.shuffle(idx_list)
        for i, idx in enumerate(idx_list):
            if i % 5000 == 0 or i == n_samples - 1:
                pct = (i + 1) / n_samples
                sys.stdout.write(f"\rEp {ep:02d}/{epochs} |{'#'*int(15*pct)+'-'*int(15-15*pct)}| {pct:>4.1%}")
                sys.stdout.flush()
            ctx, target = ids[idx:idx+5], ids[idx+5]
            h = np.mean(brain.W1[ctx], axis=0).reshape(1, -1)
            logits = np.dot(h, brain.W2)
            probs = np.exp(logits - np.max(logits)); probs /= np.sum(probs)
            probs[0, target] -= 1
            brain.W2 -= lr * np.dot(h.T, probs)
            brain.W1[ctx] -= lr * np.dot(probs, brain.W2.T) / 5
        print(f" | {time.time()-start:.1f}s")
    brain.save()

def mars_generate(brain, words, max_len=32):
    ctx_words = words[-5:]
    ids = [brain.w2i[w] for w in ctx_words if w in brain.w2i]
    if not ids or brain.W1.size == 0: return "Thinking..."
    
    res_text = []
    generated_ids = []
    
    for _ in range(max_len):
        h = np.mean(brain.W1[ids], axis=0).reshape(1, -1)
        logits = np.dot(h, brain.W2)[0]
        
        # --- ANTI-REPETITION LOGIC (PHRASE & WORD) ---
        # 1. Khởi tạo hình phạt cho từng từ
        penalties = np.zeros_like(logits)
        
        # 2. Phạt lặp từ đơn (Word-level)
        counts = Counter(generated_ids)
        for wid, count in counts.items():
            # Phạt nặng nếu lặp quá 2 lần
            penalties[wid] += (count * 5.0) if count >= 2 else (count * 2.0)

        # 3. Phạt lặp cụm từ (N-gram level: 2 to 4)
        if len(generated_ids) >= 2:
            for n in range(2, 5):
                if len(generated_ids) < n: continue
                # Lấy cụm n-gram cuối cùng vừa sinh ra
                last_ngram = tuple(generated_ids[-n:])
                # Kiểm tra xem cụm này đã xuất hiện trước đó chưa
                for i in range(len(generated_ids) - n):
                    prev_ngram = tuple(generated_ids[i:i+n])
                    if last_ngram == prev_ngram:
                        # Nếu cụm từ đã xuất hiện, phạt rất nặng từ tiếp theo để lái sang hướng khác
                        penalties += 10.0 
        
        # Áp dụng hình phạt và TF-IDF
        for idx in range(len(logits)):
            word = brain.i2w[idx]
            logits[idx] += brain.tfidf.get(word, 0) * 0.05 # TF-IDF Boost
            logits[idx] -= penalties[idx]

        # --- SAMPLING ---
        logits = np.nan_to_num(logits, nan=-10.0)
        exp_logits = np.exp((logits - np.max(logits)) / 0.8)
        probs = exp_logits / (np.sum(exp_logits) + 1e-9)
        
        # Nucleus Sampling (Top-P)
        sorted_idx = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[sorted_idx])
        probs[sorted_idx[np.where(cum_probs > 0.9)[0][0] + 1:]] = 0
        probs /= (np.sum(probs) + 1e-9)
        
        try:
            next_id = np.random.choice(len(probs), p=probs)
        except:
            next_id = np.argmax(logits)
            
        word = brain.i2w[next_id]
        if word == ".": break
        res_text.append(word)
        generated_ids.append(next_id)
        
        ids.append(next_id)
        if len(ids) > 5: ids.pop(0)
        
    return " ".join(res_text) + "."

def main():
    brain = MarsBrain()
    print("\n--- MARS-1.2 PRO (Anti-Repeat) ---")
    while True:
        m = input("Mars (T)rain / (C)hat / (E)xit: ").lower().strip()
        if m in ['t', 'c', 'e']: break
    
    if m == 't': train_mars(brain)
    elif m == 'c':
        print("\n[Mars-1.2 Online]")
        while True:
            try:
                inp = input("\nYou: ").lower().strip()
                if not inp or inp == 'exit': break
                with open(DATA_FILE, "a", encoding="utf-8") as f: f.write(inp + " .\n")
                w = inp.split(); brain.expand(w)
                if brain.W1.size == 0: print("Please train first."); continue
                print("AI: " + mars_generate(brain, w))
            except KeyboardInterrupt: break
        brain.save()
    print("\n[!] Closed.")

if __name__ == "__main__": main()
