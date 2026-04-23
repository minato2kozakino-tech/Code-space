from chat.engine import mars_generate
import os

def start_chat(brain, cfg):
    data_file = cfg['paths']['data_file']
    project_name = cfg['project']['name']
    print(f"[{project_name} Online. Type 'exit' to quit]")
    
    while True:
        try:
            inp = input("You: ").lower().strip()
            if not inp or inp == 'exit': break
            
            with open(data_file, "a", encoding="utf-8") as f:
                f.write(inp + " ." + os.linesep)
            
            words = inp.split()
            brain.expand(words) # Expand vocabulary with user input
            
            # Kiểm tra bộ não có sẵn sàng chưa
            if brain.W_in is None or brain.W_in.size == 0:
                print("AI: Please train me first. Model (Gaton) is not initialized.")
                continue
            
            response = mars_generate(brain, words, cfg)
            print("AI: " + response)
            
        except KeyboardInterrupt: break
    
    brain.save()
    print("[!] Knowledge saved. Goodbye!")
