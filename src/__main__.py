import argparse
import json
from sys import exit
from typing import List, Dict, Any
import llm_sdk
import numpy as np


def call_llm(llm: Any, full_prompt: str) -> str:
    tokens = llm.encode(full_prompt)
    tokens_list = tokens[0].tolist()
    initial_tokens_len = len(tokens_list)
    i = 100
    while i:
        logits = llm.get_logits_from_input_ids(tokens_list)
        next_token = np.argmax(logits)
        tokens_list.append(next_token)
        try:
            current_answer = llm.decode(tokens_list[initial_tokens_len - 1:])
            json.loads(current_answer)
            break
        except Exception:
            pass
        i -= 1
    result_tokens = tokens_list[initial_tokens_len - 1:]
    result_string: str = llm.decode(result_tokens)
    return result_string


def generate_outfile(functions_definition: List[Dict[str, str]],
                     input: List[Dict[str, str]]) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    fd_string = json.dumps(functions_definition)
    for prompt in input:
        prompt_string = prompt["prompt"]
        full_prompt = (f"Pick a function matching the question "
                       f"'{prompt_string}' out of the following: "
                       f"{fd_string} and return only a JSON containing "
                       f"prompt, name and the parameters:\n{{")
        result_string = call_llm(llm, full_prompt)
        print(result_string)


def parse_infile(path: str) -> List[Dict[str, str]]:
    try:
        with open(path) as f:
            infile: List[Dict[str, str]] = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"Error while parsing {path}: {e}")
        exit(1)
    return infile


def main() -> None:
    """Run the main program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="path for the functions definition JSON file")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="path for the input JSON file")
    parser.add_argument("--output",
                        default="data/output/function_calls.json",
                        help="path for the output JSON file")
    args = parser.parse_args()
    functions_definition = parse_infile(args.functions_definition)
    input = parse_infile(args.input)
    generate_outfile(functions_definition, input)


if __name__ == "__main__":
    main()
