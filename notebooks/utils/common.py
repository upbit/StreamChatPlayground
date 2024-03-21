import os

def load_plaintext(path, filename):
    with open(os.path.join(path, filename), 'r', encoding="UTF-8") as f:
        return f.read()
