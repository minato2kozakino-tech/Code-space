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
    
    if not ids or brain.W_in.size == 0:
        return "..."
    
    res_text = []
    generated_ids = []
    
    # Classify intent/language using scikit-learn classifier
    input_text = " ".join(words)
    predicted_lang = "general"
    if brain.vectorizer and brain.lang_classifier:
        input_vector = brain.vectorizer.transform([input_text])
        predicted_lang = brain.lang_classifier.predict(input_vector)[0]
        print(f"[Debug] Predicted Language/Intent: {predicted_lang}")

    # Generate Lua code if classified as 'lua'
    if predicted_lang == "lua":
        lua_generator = LuaCodeGenerator(cfg) # Initialize LuaCodeGenerator
        return lua_generator.generate_code_block(num_lines=random.randint(5, 10)) + "."

    for _ in range(max_len):
        # Forward Pass with Bias
        h1 = np.mean(brain.W_in[ids], axis=0).reshape(1, -1)
        z2 = np.dot(h1, brain.W_h) + brain.b_h
        h2 = relu(z2)
        logits = (np.dot(h2, brain.W_out) + brain.b_out)[0]
        
        # Repetition Penalty (Stronger)
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
        
        # TF-IDF Boost (using scikit-learn vectorizer if available)
        if cfg['brain']['use_tfidf'] and brain.vectorizer and brain.tfidf:
            current_context = " ".join(res_text[-5:] + [brain.i2w.get(ids[-1], '')]) # Last 5 generated words + current word
            if current_context:
                # Use vectorizer to get TF-IDF features for context
                context_vector = brain.vectorizer.transform([current_context])
                # Get top N features and boost corresponding words in logits
                feature_names = brain.vectorizer.get_feature_names_out()
                sorted_tfidf_indices = context_vector.toarray().argsort()[0][::-1]
                
                for feat_idx in sorted_tfidf_indices[:10]: # Boost top 10 relevant words
                    word = feature_names[feat_idx]
                    if word in brain.w2i:
                        logits[brain.w2i[word]] += cfg['tfidf']['boost_factor'] * context_vector[0, feat_idx]

        # Markov Chain integration (optional, if neural network still struggles)
        if cfg['brain']['use_markov'] and brain.markov_model and len(ids) >= brain.markov_model.order:
            markov_context = tuple(brain.i2w.get(i, '<unk>') for i in ids[-brain.markov_model.order:])
            next_word_by_markov = brain.markov_model.generate_next_word(markov_context)
            if next_word_by_markov in brain.w2i:
                logits[brain.w2i[next_word_by_markov]] += 5.0 # Boost Markov's suggestion

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
