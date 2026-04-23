import json
import os
import random
import sys

# Thêm đường dẫn cho LuaCodeGenerator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_generate import LuaCodeGenerator

class GrammarMars:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db_path_en = cfg['paths']['db_config_en']
        self.db_path_vn = cfg['paths']['db_config_vn']
        self.db_path_lua = cfg['paths']['db_config_lua']
        self.code_db_path = cfg['paths']['code_data_config']
        
        self.db_en = self._load_db(self.db_path_en)
        self.db_vn = self._load_db(self.db_path_vn)
        self.db_lua = self._load_db(self.db_path_lua)
        self.code_db = self._load_db(self.code_db_path)
        self.lua_generator = LuaCodeGenerator(cfg) # Initialize Lua code generator

    def _load_db(self, path):
        # Base dir for data files is the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, path)
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get(self, key, lang='en'): 
        db = self.db_en
        if lang == 'vn': db = self.db_vn
        elif lang == 'lua': db = self.db_lua
        return random.choice(db.get(key, ['unknown']))

    # --- ENGLISH GRAMMAR PATTERNS ---
    def en_simple(self): return f"{self.get('subject')} {self.get('verb')} {self.get('object')}"
    def en_opinion(self): return f"{self.get('subject')} {self.get('verb')} that {self.get('object')} is {self.get('adjective')}"
    def en_offer(self): return f"if {self.get('subject')} {self.get('verb')} {self.get('object')} , {self.get('subject')} {self.get('modal')} {self.get('verb')} {self.get('object')}"
    def en_complex_action(self): return f"{self.get('subject')} {self.get('modal')} {self.get('verb')} the {self.get('object')} {self.get('adverb')}"
    def en_conversation(self): return f"{self.get('filler')} , {self.get('subject')} {self.get('adverb')} {self.get('verb')} {self.get('preposition')} {self.get('object')}"
    def en_critical(self): return f"{self.get('subject')} {self.get('past')} {self.get('verb')} {self.get('object')}"
    def en_tech_context(self): return f"{self.get('subject')} {self.get('verb')} {self.get('tech_term')} {self.get('adverb')}"
    def en_interaction(self): return f"{self.get('greetingword')} , {self.get('firstquestion')}"
    def en_python_code(self): return f"{self.get('python_code')}"
    def en_exclamation(self): return f"{self.get('exclamation')}"

    # --- VIETNAMESE GRAMMAR PATTERNS (NÂNG CẤP) ---
    def vn_simple(self): return f"{self.get('subject_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('object_vn', 'vn')}"
    def vn_adjective(self): return f"{self.get('subject_vn', 'vn')} {self.get('tobe_vn', 'vn')} {self.get('adjective_vn', 'vn')}"
    def vn_complex(self): return f"{self.get('subject_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('object_vn', 'vn')} {self.get('adverb_vn', 'vn')}"
    def vn_question(self): return f"{self.get('greeting_vn', 'vn')} , {self.get('firstquestion_vn', 'vn')}"
    def vn_opinion(self): return f"{self.get('subject_vn', 'vn')} {self.get('filler_vn', 'vn')} {self.get('verb_vn', 'vn')} rằng {self.get('object_vn', 'vn')} {self.get('tobe_vn', 'vn')} {self.get('adjective_vn', 'vn')}"
    def vn_conditional(self): return f"nếu {self.get('subject_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('object_vn', 'vn')} thì {self.get('subject_vn', 'vn')} {self.get('modal_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('adverb_vn', 'vn')}"
    def vn_tech(self): return f"{self.get('subject_vn', 'vn')} {self.get('modal_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('tech_term_vn', 'vn')} một cách {self.get('adjective_vn', 'vn')}"
    def vn_interaction_deep(self): return f"{self.get('interaction_vn', 'vn')} , {self.get('filler_vn', 'vn')} {self.get('subject_vn', 'vn')} {self.get('verb_vn', 'vn')} {self.get('object_vn', 'vn')}"

    # --- LUA CODE GENERATION PATTERNS ---
    def lua_simple_line(self): return self.lua_generator.generate_simple_line()
    def lua_function(self): return self.lua_generator.generate_function()
    def lua_conditional(self): return self.lua_generator.generate_conditional()
    def lua_loop(self): return self.lua_generator.generate_loop()
    def lua_code_example(self): return random.choice(self.code_db['code_examples_lua'])['code']

    def build(self, languages, size_kb=10000):
        data_file = self.cfg['paths']['data_file']
        target_bytes = size_kb * 1024
        print(f"[+] Generating Multi-language Dataset for Origon AI ({size_kb}KB)...")
        
        en_patterns = [self.en_simple, self.en_opinion, self.en_offer, self.en_complex_action, self.en_conversation, self.en_critical, self.en_tech_context, self.en_interaction, self.en_python_code, self.en_exclamation]
        vn_patterns = [self.vn_simple, self.vn_adjective, self.vn_complex, self.vn_question, self.vn_opinion, self.vn_conditional, self.vn_tech, self.vn_interaction_deep]
        lua_patterns = [self.lua_simple_line, self.lua_function, self.lua_conditional, self.lua_loop, self.lua_code_example]

        all_patterns_with_weights = []
        lang_weights = self.cfg['data_gen']['language_weights']

        for lang in languages:
            if lang == 'en':
                for _ in range(int(len(en_patterns) * (lang_weights.get('en', 0) * 10))):
                    all_patterns_with_weights.append(random.choice(en_patterns))
            elif lang == 'vn':
                for _ in range(int(len(vn_patterns) * (lang_weights.get('vn', 0) * 10))):
                    all_patterns_with_weights.append(random.choice(vn_patterns))
            elif lang == 'lua':
                for _ in range(int(len(lua_patterns) * (lang_weights.get('lua', 0) * 10))):
                    all_patterns_with_weights.append(random.choice(lua_patterns))

        if not all_patterns_with_weights:
            all_patterns_with_weights = en_patterns + vn_patterns + lua_patterns

        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_file), "w", encoding="utf-8") as f:
            curr_size = 0
            while curr_size < target_bytes:
                sentence = random.choice(all_patterns_with_weights)()
                line_to_write = sentence + os.linesep + os.linesep
                f.write(line_to_write)
                curr_size += len(line_to_write.encode('utf-8'))
        print(f"[!] Dataset generation complete at {data_file}.")
