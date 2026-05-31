import argparse
import requests
import sys
import os
import yaml
import pickle
import torch
import numpy as np
import datetime

# Import our new modules
from core.rnn_model import OrigonGRU
from core.intent_classifier import IntentClassifier
from core.retriever import ContextRetriever
from core.code_handler import CodeHandler

def load_config():
    try:
        with open(os.path.join(os.path.dirname(__file__), "config/conf.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[-] Error loading config: {e}")
        return None

def main_cli():
    parser = argparse.ArgumentParser(description="Origon AI CLI Client")
    parser.add_argument("-m", "--model", default="gaton-v2", help="Model name to use (e.g., gaton-v2)")
    parser.add_argument("text", help="Message to send to AI")
    parser.add_argument("--url", default="http://localhost:5000", help="Server URL")

    args = parser.parse_args()
    cfg = load_config()
    if cfg is None:
        sys.exit(1)
    
    device = torch.device("cpu") # CLI client runs on CPU

    # Load components needed for chat
    try:
        # Load model bundle
        model_path = cfg['paths']['gaton_model']
        if not os.path.exists(model_path):
            print(f"[-] Error: Model {model_path} not found. Please train it first.")
            sys.exit(1)
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

        # Load other components
        classifier = IntentClassifier(cfg)
        classifier.load() # Load trained classifier
        retriever = ContextRetriever(cfg)
        retriever.load_index() # Load trained retriever index
        code_handler = CodeHandler(cfg)

        # Process user query
        input_text = args.text.lower()
        
        # 1. Intent Classification
        intent = classifier.predict(input_text)
        
        # 2. Routing
        if intent == "code":
            response = code_handler.handle(input_text)
        else:
            # 3. Conversational Pipeline
            # 3.1 Retrieval
            context = retriever.retrieve(input_text)
            
            # 3.2 Generation with GRU
            full_input = context + " " + input_text
            input_ids = [w2i.get(w, 0) for w in full_input.split()]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            res_text = []
            hidden = None
            
            max_len = cfg['chat']['max_len']
            curr_input = input_tensor
            
            with torch.no_grad():
                for _ in range(max_len):
                    output, hidden = model(curr_input, hidden)
                    logits = output[0, -1, :]
                    
                    # Repetition Penalty
                    for wid in [w2i.get(w, 0) for w in res_text]:
                        logits[wid] -= cfg['chat']['repetition_penalty']['base']
                    
                    # Sampling
                    probs = torch.softmax(logits / cfg['chat']['temperature'], dim=-1)
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
            
            if res_text:
                response = " ".join(res_text).capitalize() + "."
            else:
                response = "..."

        # Append to log file
        try:
            log_path = os.path.join(os.path.dirname(__file__), 'output.log')
            with open(log_path, 'a', encoding='utf-8') as lf:
                ts = datetime.datetime.utcnow().isoformat()
                lf.write(f"{ts} | model={args.model} | prompt={args.text} | response={response}\n")
        except Exception as e:
            print(f"[-] Warning: could not write log: {e}")

        print(f"AI ({args.model}): {response}")
            
    except Exception as e:
        print(f"[-] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_cli()
