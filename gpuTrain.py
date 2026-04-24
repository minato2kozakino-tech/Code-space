import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import yaml
import time
import pickle
import sys
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Thêm đường dẫn để nhận diện các mô-đun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class GatonNet(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(GatonNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Kiến trúc Deep Gaton: H1 (hidden) -> H2 (hidden) -> Out (vocab)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = torch.mean(self.embedding(x), dim=1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return self.fc3(h)

class TextDataset(Dataset):
    def __init__(self, token_ids, n_gram):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.n_gram = n_gram
    def __len__(self):
        return len(self.token_ids) - self.n_gram
    def __getitem__(self, idx):
        return self.token_ids[idx : idx + self.n_gram], self.token_ids[idx + self.n_gram]

class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.lookup = defaultdict(Counter)
        self.starts = []
    def fit(self, sentences):
        for s in sentences:
            w = s.split()
            if not w: continue
            self.starts.append(w[0])
            for i in range(len(w) - self.order):
                ctx, nxt = tuple(w[i:i+self.order]), w[i+self.order]
                self.lookup[ctx][nxt] += 1
        return self
    def generate_next_word(self, context):
        if context not in self.lookup: return np.random.choice(self.starts) if self.starts else "..."
        w, p = zip(*self.lookup[context].items())
        p = np.array(p, dtype=np.float32); p /= p.sum()
        return np.random.choice(w, p=p)

def load_config():
    with open("config/conf.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training on: {device}")

    # Load Data
    with open(cfg['paths']['data_file'], "r", encoding="utf-8") as f:
        lines = [l.strip().lower() for l in f if l.strip()]
    
    all_tokens = []
    for l in lines: all_tokens.extend(l.split())
    vocab = sorted(list(set(all_tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    token_ids = [w2i[t] for t in all_tokens]

    # CPU fitting
    vectorizer = TfidfVectorizer(max_features=5000).fit(lines)
    labels = []
    for l in lines:
        if "function" in l or "local" in l: labels.append("lua")
        elif any(ord(c) > 127 for c in l): labels.append("vn")
        else: labels.append("en")
    classifier = MultinomialNB().fit(vectorizer.transform(lines), labels)
    markov = MarkovChain(order=cfg['markov_chain']['order']).fit(lines)

    # Model & Dataloader
    model = GatonNet(len(vocab), cfg['brain']['hidden_size']).to(device)
    dataloader = DataLoader(TextDataset(token_ids, cfg['trainer']['n_gram']), batch_size=cfg['trainer']['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    print(f"[+] Starting Training Loop...")
    try:
        for ep in range(1, cfg['trainer']['epochs'] + 1):
            ep_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                if i % 500 == 0: print(f"\rEpoch {ep} | Loss: {loss.item():.4f}", end="")
            print(f"\n[!] Epoch {ep} Done. Avg Loss: {ep_loss/len(dataloader):.4f}")
    except KeyboardInterrupt: print("\n[!] Interrupted.")

    # Save
    print("[+] Exporting model...")
    sd = model.cpu().state_dict()
    save_data = {
        "W_in": sd['embedding.weight'].numpy(),
        "W_h1": sd['fc1.weight'].numpy().T, "b_h1": sd['fc1.bias'].numpy(),
        "W_h2": sd['fc2.weight'].numpy().T, "b_h2": sd['fc2.bias'].numpy(),
        "W_out": sd['fc3.weight'].numpy().T, "b_out": sd['fc3.bias'].numpy(),
        "w2i": w2i, "i2w": i2w,
        "vectorizer": vectorizer, "lang_classifier": classifier, "markov_model": markov
    }
    with open(cfg['paths']['gaton_model'], "wb") as f:
        pickle.dump(save_data, f)
    print(f"[!] Saved to {cfg['paths']['gaton_model']}")

if __name__ == "__main__":
    train()
