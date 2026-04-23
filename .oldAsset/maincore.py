import numpy as np
import os
import pickle
import sys

# Tạo thư mục lưu trữ model
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_v1.origon")
DATA_FILE = "data_300kb.txt"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ==========================================
# 1. XỬ LÝ DỮ LIỆU & N-GRAM (1n - 5n)
# ==========================================
def load_and_tokenize():
    if not os.path.exists(DATA_FILE):
        print(f"[-] Lỗi: Cần file {DATA_FILE} để luyện công!")
        sys.exit()
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        tokens = f.read().lower().split()
    
    vocab = sorted(list(set(tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    
    X_ids, y_ids = [], []
    # Quét ngữ cảnh từ 1n đến 5n
    for n in range(1, 6):
        for i in range(len(tokens) - n):
            context = [w2i[tokens[j]] for j in range(i, i + n)]
            target = w2i[tokens[i + n]]
            X_ids.append(context)
            y_ids.append(target)
            
    return X_ids, y_ids, w2i, i2w, len(vocab)

# ==========================================
# 2. KHỞI TẠO & HUẤN LUYỆN (NUMPY CORE)
# ==========================================
def train_ai(X, y, vocab_size):
    hidden_size = 8  # Tăng lên 8 để chứa được nhiều ngữ cảnh hơn
    W1 = np.random.randn(vocab_size, hidden_size) * 0.1
    W2 = np.random.randn(hidden_size, vocab_size) * 0.1
    
    lr = 0.05
    epochs = 20 # Bạn có thể tăng lên nếu máy chạy nhanh
    
    print(f"[+] Đang luyện {len(X)} mẫu ngữ cảnh (1n-5n)...")
    
    for epoch in range(epochs):
        for idx in range(len(X)):
            ctx = X[idx]
            tgt = y[idx]
            
            # Forward: Tính trung bình embedding của ngữ cảnh
            # h = (1/n) * sum(W1[ids])
            h = np.mean(W1[ctx], axis=0).reshape(1, -1)
            
            # Softmax đơn giản hóa
            scores = np.dot(h, W2)
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            
            # Backprop
            d_scores = probs
            d_scores[0, tgt] -= 1
            
            dW2 = np.dot(h.T, d_scores)
            d_h = np.dot(d_scores, W2.T)
            
            # Cập nhật
            W2 -= lr * dW2
            W1[ctx] -= lr * d_h / len(ctx)
            
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}/{epochs} hoàn tất...")
            
    return W1, W2

# ==========================================
# 3. GIAO DIỆN CHAT & LƯU TRỮ
# ==========================================
def main():
    # Bước 1: Train
    X, y, w2i, i2w, v_size = load_and_tokenize()
    W1, W2 = train_ai(X, y, v_size)
    
    # Bước 2: Đóng gói vào định dạng .origon
    brain_data = {
        "W1": W1, "W2": W2, "w2i": w2i, "i2w": i2w, "v_size": v_size
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(brain_data, f)
    print(f"[+] Đã xuất xưởng model tại: {MODEL_PATH}")

    # Bước 3: Chat trực tiếp
    print("\n" + "="*30)
    print(" AI CHATBOT (1n-5n Context) ")
    print(" Gõ 'exit' để nghỉ tay. ")
    print("="*30)

    while True:
        user_msg = input("Bạn: ").lower().strip().split()
        if not user_msg: continue
        if user_msg[0] == 'exit': break
        
        # Lấy tối đa 5 từ cuối làm ngữ cảnh để dự đoán
        ctx_ids = [w2i[w] for w in user_msg[-5:] if w in w2i]
        
        if not ctx_ids:
            print("AI: Chịu, từ này tôi chưa học.")
            continue
            
        h = np.mean(W1[ctx_ids], axis=0).reshape(1, -1)
        res = np.dot(h, W2)
        ans_id = np.argmax(res)
        print(f"AI: {i2w[ans_id]}")

if __name__ == "__main__":
    main()

