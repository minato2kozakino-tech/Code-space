import os
import sys
import yaml
import pandas as pd

# Thêm đường dẫn để nhận diện các mô-đun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core.brain import MarsBrain
from core.trainer import train_mars
# from core.data_gen import GrammarMars # REMOVED: No longer used for generation
from chat.interface import start_chat
from gpuTrain import train_gpu # Import GPU training script

def load_config():
    config_path = os.path.join(BASE_DIR, "config/conf.yaml")
    if not os.path.exists(config_path):
        print(f"[-] Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    
    # Đảm bảo các thư mục tồn tại
    os.makedirs(cfg['paths']['model_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(cfg['paths']['conversational_data']), exist_ok=True) # Ensure database dir exists
    os.makedirs(os.path.dirname(cfg['paths']['gaton_model']), exist_ok=True) # Ensure gaton model dir exists
    os.makedirs(os.path.dirname(cfg['paths']['intent_model']), exist_ok=True)

    # Khởi tạo não bộ AI (chỉ để truyền config cho các module khác)
    # Brain object will be loaded with actual components in chat/trainer
    brain = MarsBrain(cfg)

    # Định dạng menu chính
    menu = f"""
================================================================
 {cfg['project']['name']} v{cfg['project']['version']} 
 [D] Prepare Data   | [T] Train (CPU)      | [N] Train (GPU) 
 [C] Chat           | [S] Server           | [E] Exit 
================================================================
"""
    print(menu)

    try:
        choice = input("Choice: ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        choice = 'e'

    if choice == 'd': # New option to prepare conversational data
        csv_path = cfg['paths']['conversational_data']
        if os.path.exists(csv_path):
            print(f"[+] Found existing conversational data at {csv_path}.")
            df = pd.read_csv(csv_path)
            print(f"[+] Loaded {len(df)} conversational pairs.")
        else:
            print(f"[-] Conversational data not found at {csv_path}.")
            print("[*] Please create a CSV file with 'human' and 'bot' columns.")
            print("[*] Example: human,bot
"hello","hi!"")
        print("[!] Data preparation check complete.")
    
    elif choice == 't':
        print("[+] Starting CPU Training (using current model paths)...")
        train_mars(brain, cfg) # This will be adapted to train the new architecture
        print("[!] Training process finished.")

    elif choice == 'n':
        print("[+] Starting GPU Training (Requires PyTorch + CUDA)...")
        train_gpu() # Call the new GPU training script
        print("[!] GPU Training process finished.")
    
    elif choice == 'c' or choice == '':
        start_chat(brain, cfg)

    elif choice == 's':
        print("[+] Starting Origon AI Server...")
        server_path = os.path.join(BASE_DIR, 'server/app.py')
        os.system(f"python3 {server_path}")

    elif choice == 'e':
        print("[!] Closing system. Goodbye!")

if __name__ == "__main__":
    main()
