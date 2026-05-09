import json
import os
import random

class CodeHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lua_db = self._load_json(cfg['paths']['code_db_lua'])
        self.python_db = self._load_json(cfg['paths']['code_db_python'])

    def _load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[-] Error loading JSON from {path}: {e}")
                return {}
        return {}

    def handle(self, query):
        query = query.lower()
        # Simple detection for Python vs Lua
        if "python" in query or "py" in query:
            return self.generate_python(query)
        else:
            return self.generate_lua(query)

    def generate_lua(self, query):
        if not self.lua_db: return "[-] Lua code database not found."
        
        # Heuristic matching
        snippets = self.lua_db.get('common_snippets_lua', [])
        patterns = self.lua_db.get('syntax_patterns_lua', [])
        
        if "function" in query or "hàm" in query:
            return random.choice([p for p in patterns if "function" in p] or snippets)
        if "if" in query or "nếu" in query:
            return random.choice([p for p in patterns if "if" in p] or snippets)
        
        return random.choice(snippets) if snippets else "print('Hello Lua!')"

    def generate_python(self, query):
        # We'll need a similar python_data.json later
        if not self.python_db: 
            return "print('Hello Python!\\n# Python support is being enhanced.')"
        
        snippets = self.python_db.get('common_snippets_py', [])
        return random.choice(snippets) if snippets else "print('Python snippet')"
