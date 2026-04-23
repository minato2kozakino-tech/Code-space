import json
import os
import random

class GrammarMars:
    def __init__(self, db_path="database/en_data.json"):
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        self.db = db

    def get(self, key): return random.choice(self.db.get(key))

    def opinion(self):
        # I think that AI is smart
        return f"{self.get('subject')} {self.get('verb')} that {self.get('object')} is {self.get('adjective')}"

    def offer(self):
        # If you need code , I can help you
        return f"if you need {self.get('object')} , {self.get('subject')} {self.get('modal')} {self.get('verb')} {self.get('object')}"

    def complex_action(self):
        # We must optimize the system efficiently
        return f"{self.get('subject')} {self.get('modal')} {self.get('verb')} the {self.get('object')} {self.get('adverb')}"

    def conversation(self):
        # Well, I really believe in logic
        return f"{self.get('filler')} , {self.get('subject')} {self.get('adverb')} {self.get('verb')} in {self.get('object')}"

    def simple(self):
        return f"{self.get('subject')} {self.get('verb')} {self.get('object')}"

    def build(self, size_kb=2000):
        target_bytes = size_kb * 1024
        print(f"[+] Generating Mars-1 Dataset ({size_kb}KB)...")
        patterns = [self.opinion, self.offer, self.complex_action, self.conversation, self.simple]
        with open("data_300kb.txt", "w", encoding="utf-8") as f:
            curr_size = 0
            while curr_size < target_bytes:
                line = random.choice(patterns)() + " .\n"
                f.write(line)
                curr_size += len(line.encode('utf-8'))
        print(f"[!] Mars-1 data ready.")

if __name__ == "__main__":
    GrammarMars().build(2000)
