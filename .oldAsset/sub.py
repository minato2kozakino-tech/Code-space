import numpy as np
import random
import sys

# ==========================================
# 1. TẠO TỪ VỰNG & DỮ LIỆU HỘI THOẠI
# ==========================================
# Thêm một ít từ chào hỏi để có cảm giác "nói chuyện"
subjects = ["i", "you", "we", "they"]
verbs = ["love", "drink", "code", "eat", "see", "can", "know", "don't"]
objects = ["water", "programming", "bread", "you", "it", "python", "termux"]
greetings = ["hi", "hello", "hey"]

all_words = list(set(subjects + verbs + objects + greetings + ["fine", "bot"]))
word_to_id = {word: i for i, word in enumerate(all_words)}
id_to_word = {i: word for word, i in word_to_id.items()}
vocab_size = len(all_words)

# Tạo 5000 mẫu hội thoại giả lập (~300KB)
dataset = []
for _ in range(5000):
    case = random.random()
    if case < 0.7: # 70% là câu S-V-O
        s, v, o = random.choice(subjects), random.choice(verbs), random.choice(objects)
        dataset.append([word_to_id[s], word_to_id[v], word_to_id[o]])
    else: # 30% là chào hỏi
        g = random.choice(greetings)
        dataset.append([word_to_id[g], word_to_id[g], word_to_id["hi"]]) # Hi -> Hi

data = np.array(dataset)
X = data[:, :2] 
y = data[:, 2]

# ==========================================
# 2. KHỞI TẠO WEIGHTS (150 THAM SỐ)
# ==========================================
hidden_size = 4
W1 = np.random.randn(vocab_size, hidden_size) * 0.1
W2 = np.random.randn(hidden_size, vocab_size) * 0.1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ==========================================
# 3. HUẤN LUYỆN (QUY TRÌNH TRAIN)
# ==========================================
print("--- AI đang học cách nói chuyện... ---")
lr = 0.1
for epoch in range(101):
    hidden = np.mean(W1[X], axis=1)
    probs = softmax(np.dot(hidden, W2))
    
    # Tính lỗi và cập nhật (Gradients)
    dscores = probs
    dscores[range(len(y)), y] -= 1
    dscores /= len(y)
    
    dW2 = np.dot(hidden.T, dscores)
    dhidden = np.dot(dscores, W2.T)
    
    W2 -= lr * dW2
    for i in range(len(X)):
        W1[X[i]] -= lr * dhidden[i] / 2
        
    if epoch % 50 == 0:
        loss = -np.mean(np.log(probs[range(len(y)), y]))
        print(f"Tiến độ: {epoch}% - Độ hiểu: {100 - loss*100:.2f}%")

# ==========================================
# 4. GIAO DIỆN NÓI CHUYỆN (CHAT LOOP)
# ==========================================
print("\n" + "="*30)
print(" AI ĐÃ SẴN SÀNG NÓI CHUYỆN! ")
print("(Nhập 'exit' để thoát)")
print("="*30)

while True:
    user_input = input("Bạn: ").lower().strip()
    
    if user_input == 'exit':
        break
        
    words = user_input.split()
    
    # Xử lý câu nhập vào
    if len(words) < 1:
        continue
        
    try:
        # Lấy ID của các từ người dùng nhập (nếu có trong từ điển)
        input_ids = [word_to_id[w] for w in words if w in word_to_id]
        
        if len(input_ids) == 0:
            print("AI: Hả? Tôi chưa học từ đó...")
            continue
            
        # AI tính toán phản hồi
        h = np.mean(W1[input_ids], axis=0)
        res = np.dot(h, W2)
        out_id = np.argmax(res)
        
        print(f"AI: {id_to_word[out_id]}")
        
    except Exception:
        print("AI: Tôi hơi lag, nói lại đi!")

print("Tạm biệt!")

