import argparse
import json
from sys import exit
from typing import List, Dict


def parse_infile(path: str) -> List[Dict[str, str]]:
    try:
        with open(path) as f:
            infile: List[Dict[str, str]] = json.loads(f.read())
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"Error while parsing {path}: {e}")
        exit(1)
    return infile


def main() -> None:
    """Run the main program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="path for the functions definition json file")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="path for the input json file")
    parser.add_argument("--output",
                        default="data/output/function_calls.json",
                        help="path for the output json file")
    args = parser.parse_args()
    functions_definition = parse_infile(args.functions_definition)
    input = parse_infile(args.input)
    print(functions_definition, "\n\n", input)


if __name__ == "__main__":
    main()
