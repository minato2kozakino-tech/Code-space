import random
import json
import os

class LuaCodeGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        # Paths are relative to the project root (where main.py is)
        self.lua_db_path = cfg['paths']['db_config_lua']
        self.code_db_path = cfg['paths']['code_data_config']
        self.lua_db = self._load_db(self.lua_db_path)
        self.code_db = self._load_db(self.code_db_path)

    def _load_db(self, path):
        # Base dir is the parent of 'core' or the root where main.py sits
        # Since this file is in the root, we can just use the path or join with project root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
             # Fallback for when called from subdirectories
             full_path = os.path.join(base_dir, "..", path)
             
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_keyword(self): return random.choice(self.lua_db['keywords_lua'])
    def get_syntax_pattern(self): return random.choice(self.lua_db['syntax_patterns_lua'])
    def get_code_example(self): return random.choice(self.code_db['code_examples_lua'])['code']
    def get_code_pattern(self): return random.choice(self.code_db['code_patterns_lua'])

    def generate_simple_line(self):
        return random.choice([
            f"local {random.choice(['x', 'y', 'z'])} = {random.randint(1, 100)}",
            f"print('{self.get_keyword()} from generated code!')",
            f"{random.choice(['foo', 'bar'])} = {{}}"
        ])

    def generate_function(self, name="myFunc"):
        body = [self.generate_simple_line() for _ in range(random.randint(1, 3))]
        return f"function {name}(){os.linesep}  {os.linesep.join(body)}{os.linesep}end"

    def generate_conditional(self):
        condition = f"{random.choice(['x', 'y'])} > {random.randint(0, 10)}"
        true_branch = self.generate_simple_line()
        false_branch = self.generate_simple_line()
        return (
            f"if {condition} then{os.linesep}"
            f"  {true_branch}{os.linesep}"
            f"else{os.linesep}"
            f"  {false_branch}{os.linesep}"
            f"end"
        )

    def generate_loop(self):
        loop_type = random.choice(["for_numeric", "while"])
        if loop_type == "for_numeric":
            start = random.randint(1, 5)
            end = random.randint(start + 1, 10)
            body = self.generate_simple_line()
            return f"for i = {start}, {end} do{os.linesep}  {body}{os.linesep}end"
        else: # while loop
            condition = f"{random.choice(['i', 'j'])} < {random.randint(5, 15)}"
            body = self.generate_simple_line()
            return f"local i = 0{os.linesep}while {condition} do{os.linesep}  {body}{os.linesep}  i = i + 1{os.linesep}end"

    def generate_code_block(self, num_lines=5):
        code_block = []
        for _ in range(num_lines):
            choice = random.choice([
                self.generate_simple_line,
                lambda: self.generate_function(name=f"func_{random.randint(1,100)}"),
                self.generate_conditional,
                self.generate_loop
            ])
            code_block.append(choice())
        return os.linesep.join(code_block)
