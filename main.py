import os
import sys
import yaml

# Thêm đường dẫn để nhận diện các mô-đun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from core.brain import MarsBrain
from core.trainer import train_mars
from core.data_gen import GrammarMars
from chat.interface import start_chat

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
    os.makedirs(os.path.dirname(cfg['paths']['data_file']), exist_ok=True)
    os.makedirs(os.path.dirname(cfg['paths']['gaton_model']), exist_ok=True)

    # Khởi tạo não bộ AI
    brain = MarsBrain(cfg)

    # Định dạng menu chính
    menu = f"""
================================================================
 {cfg['project']['name']} v{cfg['project']['version']} 
 [G] Generate Data  | [T] Train (CPU)      | [N] Train (GPU) 
 [C] Chat           | [S] Server           | [E] Exit 
================================================================
"""
    print(menu)

    try:
        choice = input("Choice: ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        choice = 'e'

    if choice == 'g':
        valid_languages = cfg['data_gen']['languages']
        while True:
            lang_choice = input(f"Generate data in ({'/'.join(valid_languages)}/all): ").lower().strip()
            if lang_choice in valid_languages or lang_choice == 'all': break
            print(f"Invalid choice. Please enter one of {', '.join(valid_languages)} or 'all'.")
        
        generator = GrammarMars(cfg)
        if lang_choice == 'all':
            generator.build(languages=valid_languages, size_kb=cfg['data_gen']['size_kb'])
        else:
            generator.build(languages=[lang_choice], size_kb=cfg['data_gen']['size_kb'])
        print("[!] Data generation complete. You should train the model now.")
    
    elif choice == 't':
        print("[+] Starting CPU Training...")
        train_mars(brain, cfg)
        print("[!] Training process finished.")

    elif choice == 'n':
        print("[+] Starting GPU Training (Requires Torch + CUDA)...")
        gpu_script = os.path.join(BASE_DIR, "gpuTrain.py")
        if os.path.exists(gpu_script):
            os.system(f"python3 {gpu_script}")
        else:
            print("[-] Error: gpuTrain.py not found.")
    
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
