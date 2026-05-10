import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import yaml
import time
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import our new modules
from core.rnn_model import OrigonGRU
from core.intent_classifier import IntentClassifier
from core.retriever import ContextRetriever

def load_config():
    with open("config/conf.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

# --- CUSTOM DATASET FOR CONVERSATIONAL TRAINING ---
class ConversationalDataset(Dataset):
    def __init__(self, human_texts, bot_texts, w2i, max_len):
        self.samples = []
        for h, b in zip(human_texts, bot_texts):
            h_ids = [w2i.get(w, 0) for w in str(h).lower().split()]
            b_ids = [w2i.get(w, 0) for w in str(b).lower().split()]
            # Simple approach: Input is human text, target is bot text
            # Pad or truncate to max_len
            h_ids = h_ids[:max_len] + [0] * (max_len - len(h_ids))
            b_ids = b_ids[:max_len] + [0] * (max_len - len(b_ids))
            self.samples.append((torch.tensor(h_ids), torch.tensor(b_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_gpu():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training Gaton v2.0 on: {device}")

    # 1. Load CSV Data
    csv_path = cfg['paths']['conversational_data']
    if not os.path.exists(csv_path):
        print(f"[-] CSV Data not found at {csv_path}. Please create it first.")
        return
    
    df = load_conversational_csv(csv_path)
    
    if df.empty or 'bot' not in df.columns or 'human' not in df.columns:
        print("[-] CSV must have 'human' and 'bot' columns.")
        return
    
    human_texts = [str(x) for x in df['human'].dropna().tolist()]
    bot_texts = [str(x) for x in df['bot'].dropna().tolist()]

    # 2. Build Intent Classifier & Retriever
    classifier = IntentClassifier(cfg)
    # Synthetic labels for initial training if not provided in CSV
    # In a real dataset, you'd have a 'label' column.
    labels = ["chat"] * len(human_texts)
    # Detect code-related samples to label them
    for i, h in enumerate(human_texts):
        if any(w in str(h).lower() for w in ["code", "lua", "python", "hàm", "function"]):
            labels[i] = "code"
    classifier.train(human_texts, labels)

    retriever = ContextRetriever(cfg)
    retriever.build_index()

    # 3. Prepare Vocab & PyTorch Model
    all_text = " ".join(human_texts + bot_texts).lower()
    tokens = all_text.split()
    vocab = sorted(list(set(tokens)))
    w2i = {w: i + 1 for i, w in enumerate(vocab)} # 0 is for padding
    w2i["<PAD>"] = 0
    i2w = {i: w for w, i in w2i.items()}
    vocab_size = len(w2i)

    hidden_size = cfg['brain']['hidden_size']
    embedding_dim = cfg['brain']['embedding_dim']
    num_layers = cfg['brain']['num_layers']
    max_len = cfg['chat']['max_len']

    model = OrigonGRU(vocab_size, embedding_dim, hidden_size, num_layers).to(device)
    dataset = ConversationalDataset(human_texts, bot_texts, w2i, max_len)
    dataloader = DataLoader(dataset, batch_size=cfg['trainer']['batch_size'], shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print(f"[+] Training RNN for {cfg['trainer']['epochs']} epochs...")
    for ep in range(1, cfg['trainer']['epochs'] + 1):
        start_time = time.time()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            # output shape: (batch, seq_len, vocab_size)
            loss = criterion(output.transpose(1, 2), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['trainer']['gradient_clip_norm'])
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"\rEpoch {ep:03d} | Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s", end="")
        if ep % 10 == 0: print("")

    # 5. Save all as Bundle
    print("\n[+] Saving model bundle...")
    bundle = {
        "pytorch_state": model.cpu().state_dict(),
        "w2i": w2i, "i2w": i2w,
        "config": cfg
    }
    os.makedirs(os.path.dirname(cfg['paths']['gaton_model']), exist_ok=True)
    with open(cfg['paths']['gaton_model'], 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[!] Gaton v2.0 ready at {cfg['paths']['gaton_model']}")

if __name__ == "__main__":
    train_gpu()
