import csv
import pickle
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = cfg['paths']['conversational_data']
        self.index_path = cfg['paths']['retriever_data']
        self.vectorizer = TfidfVectorizer(max_features=cfg['retriever']['tfidf_max_features'])
        self.tfidf_matrix = None
        self.dataset = None

    def load_conversational_csv(self, csv_path):
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
            print(f"[!] No usable conversation rows found in {csv_path}")
            return None
        df = pd.DataFrame(cleaned, columns=['human', 'bot'])
        df['human'] = df['human'].astype(str)
        df['bot'] = df['bot'].astype(str)
        return df

    def build_index(self):
        if not os.path.exists(self.data_path):
            print(f"[-] Conversational data not found at {self.data_path}")
            return False
        
        print("[+] Building TF-IDF Index for Retrieval...")
        self.dataset = self.load_conversational_csv(self.data_path)
        if self.dataset is None or self.dataset.empty or 'human' not in self.dataset.columns:
            print("[-] Dataset is empty or missing 'human' column.")
            return False
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset['human'].astype(str))
            self.save_index()
            return True
        except Exception as e:
            print(f"[-] Error building index: {e}")
            return False

    def retrieve(self, query, top_k=None):
        if self.tfidf_matrix is None:
            return ""
        
        if top_k is None: top_k = self.cfg['retriever']['top_k']
        
        query_vec = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = similarities.argsort()[-top_k:][::-1]
        
        contexts = []
        for idx in related_indices:
            if similarities[idx] > 0.1: # Threshold to ensure relevance
                human_text = self.dataset.iloc[idx]['human']
                bot_text = self.dataset.iloc[idx]['bot']
                contexts.append(f"Human: {human_text} Bot: {bot_text}")
        
        return " ".join(contexts)

    def save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'dataset': self.dataset
            }, f)

    def load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['tfidf_matrix']
                    self.dataset = data['dataset']
                return True
            except Exception as e:
                print(f"[-] Error loading index: {e}")
                return self.build_index()
        return self.build_index()
