import re
import os

configs_code = open(r'd:\OneDrive\研究生\202603\归档\v8.4-817\ALB_RL_Project\configs.py', encoding='utf-8').read()
config_keys = re.findall(r'^[ \t]*([a-zA-Z_0-9]+)\s*=', configs_code, flags=re.MULTILINE)

project_dir = r'd:\OneDrive\研究生\202603\归档\v8.4-817\ALB_RL_Project'

for file in os.listdir(project_dir):
    if file.endswith('.py'):
        filepath = os.path.join(project_dir, file)
        code = open(filepath, encoding='utf-8').read()
        getattrs = re.findall(r'getattr\s*\(\s*configs\s*,\s*[\'"]([a-zA-Z_0-9]+)[\'"]', code)
        for attr in set(getattrs):
            status = 'OK' if attr in config_keys else 'MISSING IN CONFIGS'
            print(f'{file}: {attr} -> {status}')
