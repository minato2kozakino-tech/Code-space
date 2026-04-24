import numpy as np
from collections import Counter
import random
import os
import sys

# Thêm đường dẫn cho LuaCodeGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from code_generate import LuaCodeGenerator

def relu(x):
    return np.maximum(0, x)

def mars_generate(brain, words, cfg):
    max_len = cfg['chat']['max_len']
    temp = cfg['chat']['temperature']
    top_p = cfg['chat']['top_p']
    pen_base = cfg['chat']['repetition_penalty']['base']
    pen_inc = cfg['chat']['repetition_penalty']['incremental']
    context_window = cfg['chat']['context_window']

    ctx_words = words[-context_window:]
    ids = [brain.w2i[w] for w in ctx_words if w in brain.w2i]
    
    if not ids or brain.W_in is None or brain.W_in.size == 0:
        return "..."
    
    res_text = []
    generated_ids = []
    
    # Classify intent/language
    input_text = " ".join(words)
    predicted_lang = "general"
    if brain.vectorizer and brain.lang_classifier:
        input_vector = brain.vectorizer.transform([input_text])
        predicted_lang = brain.lang_classifier.predict(input_vector)[0]

    # Lua generation
    if predicted_lang == "lua":
        lua_generator = LuaCodeGenerator(cfg)
        return lua_generator.generate_code_block(num_lines=random.randint(5, 10)) + "."

    for _ in range(max_len):
        # Forward Pass linh hoạt
        # Layer 1 (Embedding mean)
        h = np.mean(brain.W_in[ids], axis=0).reshape(1, -1)
        
        # Layer 2 (H1)
        if brain.W_h1 is not None:
            h = relu(np.dot(h, brain.W_h1) + brain.b_h1)
            
        # Layer 3 (H2 - chỉ có ở mô hình Deep)
        if brain.W_h2 is not None:
            h = relu(np.dot(h, brain.W_h2) + brain.b_h2)
            
        # Output Layer
        logits = (np.dot(h, brain.W_out) + brain.b_out)[0]
        
        # Repetition Penalty
        counts = Counter(generated_ids)
        for wid, count in counts.items():
            logits[wid] -= (pen_base + count * pen_inc)

        # N-gram Penalty
        if len(generated_ids) >= 2:
            for n in range(2, 5):
                if len(generated_ids) < n: continue
                last_ngram = tuple(generated_ids[-n:])
                for i in range(len(generated_ids) - n):
                    if last_ngram == tuple(generated_ids[i:i+n]):
                        logits[last_ngram[-1]] -= 30.0
                        break
        
        # TF-IDF Boost
        if cfg['brain']['use_tfidf'] and brain.vectorizer:
            word_idx = len(res_text)
            # (Rút gọn logic TF-IDF để chạy nhanh hơn)
            pass

        # Markov Chain integration
        if cfg['brain']['use_markov'] and brain.markov_model and len(ids) >= brain.markov_model.order:
            markov_context = tuple(brain.i2w.get(i, '<unk>') for i in ids[-brain.markov_model.order:])
            next_word_by_markov = brain.markov_model.generate_next_word(markov_context)
            if next_word_by_markov in brain.w2i:
                logits[brain.w2i[next_word_by_markov]] += 5.0

        # Sampling
        logits = np.nan_to_num(logits, nan=-10.0)
        exp_logits = np.exp((logits - np.max(logits)) / temp)
        probs = exp_logits / (np.sum(exp_logits) + 1e-9)
        
        # Top-P
        sorted_idx = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[sorted_idx])
        cutoff = np.where(cum_probs >= top_p)[0]
        if len(cutoff) > 0:
            probs[sorted_idx[cutoff[0]+1:]] = 0
        probs /= (np.sum(probs) + 1e-9)
        
        try:
            next_id = np.random.choice(len(probs), p=probs)
        except:
            next_id = np.argmax(logits)
            
        word = brain.i2w.get(next_id, '<unk>')
        if word == ".": break
        res_text.append(word)
        generated_ids.append(next_id)
        
        ids.append(next_id)
        if len(ids) > context_window: ids.pop(0)
        
    return " ".join(res_text) + "."
