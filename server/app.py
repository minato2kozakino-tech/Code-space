import os
import sys
import yaml
from flask import Flask, request, jsonify, render_template

# Thêm thư mục gốc vào path để nạp modules core/chat
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.brain import MarsBrain
from chat.engine import mars_generate

app = Flask(__name__)

# Nạp cấu hình
def load_config():
    config_path = os.path.join(BASE_DIR, "config/conf.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
# Khởi tạo não bộ Origon (Dùng MarsBrain core)
brain = MarsBrain(cfg)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    user_text = data['text']
    model_request = data.get('model', 'mars-1') # Có thể mở rộng để nạp model khác
    
    # Sinh phản hồi từ AI
    words = user_text.lower().strip().split()
    brain.expand(words) # Tự học token mới từ API
    
    response = mars_generate(brain, words, cfg)
    
    return jsonify({
        "status": "success",
        "model": model_request,
        "response": response
    })

if __name__ == '__main__':
    # Chạy trên toàn bộ interface để có thể truy cập từ ngoài
    app.run(host='0.0.0.0', port=5000)
