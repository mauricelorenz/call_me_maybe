import json
from sys import exit
from typing import List, Dict

def parse_infile(path: str) -> List[Dict[str, str]]:
    try:
        with open(path) as f:
            infile: List[Dict[str, str]] = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"Error while parsing {path}: {e}")
        exit(1)
    return infile
