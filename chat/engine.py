import torch
import numpy as np
import pickle
import os
import random
from collections import Counter

from core.rnn_model import OrigonGRU
from core.intent_classifier import IntentClassifier
from core.retriever import ContextRetriever
from core.code_handler import CodeHandler

def mars_generate(brain, words, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = " ".join(words).lower()
    
    # 1. Intent Classification
    intent_clf = IntentClassifier(cfg)
    intent_clf.load()
    intent = intent_clf.predict(input_text)
    
    # 2. Routing
    if intent == "code":
        handler = CodeHandler(cfg)
        return handler.handle(input_text)
    
    # 3. Conversational Pipeline (Chat)
    # 3.1 Retrieval
    retriever = ContextRetriever(cfg)
    retriever.load_index()
    context = retriever.retrieve(input_text)
    
    # 3.2 Generation with GRU
    # Load bundle
    model_path = cfg['paths']['gaton_model']
    if not os.path.exists(model_path):
        return "Model not trained yet."
        
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    
    w2i = bundle['w2i']
    i2w = bundle['i2w']
    vocab_size = len(w2i)
    
    model = OrigonGRU(
        vocab_size, 
        cfg['brain']['embedding_dim'], 
        cfg['brain']['hidden_size'], 
        cfg['brain']['num_layers']
    ).to(device)
    model.load_state_dict(bundle['pytorch_state'])
    model.eval()

    # Encode input with context
    full_input = context + " " + input_text
    input_ids = [w2i.get(w, 0) for w in full_input.split()]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    res_text = []
    hidden = None
    
    # Generate tokens
    max_len = cfg['chat']['max_len']
    curr_input = input_tensor
    
    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(curr_input, hidden)
            # Take only the last token output
            logits = output[0, -1, :]
            
            # Repetition Penalty
            for wid in [w2i.get(w, 0) for w in res_text]:
                logits[wid] -= cfg['chat']['repetition_penalty']['base']
            
            # Sampling
            probs = torch.softmax(logits / cfg['chat']['temperature'], dim=-1)
            # Top-P
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_to_remove = cumulative_probs > cfg['chat']['top_p']
            idx_to_remove[..., 1:] = idx_to_remove[..., :-1].clone()
            idx_to_remove[..., 0] = 0
            probs[sorted_indices[idx_to_remove]] = 0
            probs[0] = 0
            if probs.sum() == 0:
                probs = torch.softmax(logits / cfg['chat']['temperature'], dim=-1)
                probs[0] = 0
            if probs.sum() == 0:
                break
            probs /= probs.sum()
            
            next_id = torch.multinomial(probs, 1).item()
            word = i2w.get(next_id, "")
            
            if word == "" or word == "<PAD>" or word == ".":
                fallback_ids = [idx.item() for idx in sorted_indices if idx.item() != 0 and i2w.get(idx.item(), "") not in ["", "."]]
                if not fallback_ids:
                    break
                next_id = fallback_ids[0]
                word = i2w.get(next_id, "")
            
            if word == "" or word == "<PAD>" or word == ".":
                break
                
            res_text.append(word)
            curr_input = torch.tensor([[next_id]], dtype=torch.long).to(device)

    # 4. Markov Refinement (Optional smooth)
    if not res_text: return "..."
    
    return " ".join(res_text).capitalize() + "."
