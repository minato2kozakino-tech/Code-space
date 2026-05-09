import torch
import numpy as np
import pickle
import os

from core.rnn_model import OrigonGRU
from core.intent_classifier import IntentClassifier
from core.retriever import ContextRetriever
from core.code_handler import CodeHandler

def start_chat(brain_placeholder, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg['project']['name']} Online. Type 'exit' to quit]")
    
    # Load components for chat
    try:
        model_path = cfg['paths']['gaton_model']
        if not os.path.exists(model_path):
            print(f"[-] Error: Model {model_path} not found. Please train it first.")
            return
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

        classifier = IntentClassifier(cfg)
        classifier.load() 
        retriever = ContextRetriever(cfg)
        retriever.load_index() 
        code_handler = CodeHandler(cfg)

    except Exception as e:
        print(f"[-] Error loading chat components: {e}. Please ensure models are trained.")
        return

    while True:
        try:
            inp = input("You: ").lower().strip()
            if not inp or inp == 'exit': break
            
            # --- NEW CHAT LOGIC ---
            # 1. Intent Classification
            intent = classifier.predict(inp)
            print(f"[Debug] Predicted Intent: {intent}")

            # 2. Routing
            if intent == "code":
                response = code_handler.handle(inp)
            else:
                # 3. Conversational Pipeline (Chat)
                # 3.1 Retrieval
                context = retriever.retrieve(inp)
                
                # 3.2 Generation with GRU
                full_input = context + " " + inp
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

            print("AI: " + response)
            
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"[-] An error occurred during chat: {e}")
    
    print("[!] Closing chat. Goodbye!")
