import argparse
import json
from sys import exit
from typing import List, Dict, Any
import llm_sdk
import numpy as np


def call_llm(llm: Any, full_prompt: str, vocab_json: Dict[str, int]) -> str:
    tokens = llm.encode(full_prompt)
    tokens_list = tokens[0].tolist()
    max_tokens = 100
    state = "START"
    generated = ""
    while max_tokens:
        logits = llm.get_logits_from_input_ids(tokens_list)
        if state == "START":
            logits_array = np.array(logits)
            masked = np.full(len(logits), -np.inf)
            masked[vocab_json["{"]] = logits_array[vocab_json["{"]]
            next_token = np.argmax(masked)
        else:
            next_token = np.argmax(logits)
        generated += llm.decode([next_token])
        tokens_list.append(int(next_token))
        if state == "START" and generated == "{":
            state = "NEXT"
        try:
            json.loads(generated)
            break
        except Exception:
            pass
        max_tokens -= 1
    return generated


def generate_outfile(functions_definition: List[Dict[str, str]],
                     input: List[Dict[str, str]]) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    vocab_json = parse_infile(llm.get_path_to_vocab_file())
    fd_string = json.dumps(functions_definition)
    for prompt in input:
        prompt_string = prompt["prompt"]
        full_prompt = (f"Pick a function matching the question "
                       f"'{prompt_string}' out of the following: "
                       f"{fd_string} and return only a JSON containing "
                       f"prompt, name and the parameters.")
        result_string = call_llm(llm, full_prompt, vocab_json)
        print(result_string)


def parse_infile(path: str) -> Any:
    try:
        with open(path) as f:
            infile: Any = json.load(f)
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
