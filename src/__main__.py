import argparse
import json
from sys import exit
from typing import List, Dict, Any
import llm_sdk
import numpy as np


def call_llm(llm: Any, functions_definition: List[Dict[str, str]], prompt_string: str, vocab_json: Dict[str, int]) -> str:
    fd_string = json.dumps(functions_definition)
    full_prompt = (f"Pick a function matching the question "
                f"'{prompt_string}' out of the following: "
                f"{fd_string} and return only a JSON containing "
                f"prompt, name and the parameters.")
    full_prompt_tokens = llm.encode(full_prompt)
    full_prompt_tokens_list = full_prompt_tokens[0].tolist()
    max_tokens = 100
    state = "START"
    template_list = [f"{{\"prompt\": {prompt_string}, \"name\": \"", "\", \"parameters\": {\"", "}}"]
    template_tokens = []
    for string in template_list:
        template_token = llm.encode(string)
        template_tokens.append(template_token[0].tolist())
    i = 0
    generated = []
    while max_tokens:
        logits = llm.get_logits_from_input_ids(full_prompt_tokens_list)
        if state == "START" and i < len(template_tokens[0]):
            logits_array = np.array(logits)
            masked = np.full(len(logits), -np.inf)
            masked[template_tokens[0][i]] = logits_array[template_tokens[0][i]]
            next_token = np.argmax(masked)
            i += 1
        elif state == "NAME":
            next_token = np.argmax(logits)
        generated.append(int(next_token))
        full_prompt_tokens_list.append(int(next_token))
        if state == "START" and i >= len(template_tokens[0]):
            state = "NAME"
            break
        try:
            json.loads(llm.decode(generated))
            break
        except Exception:
            pass
        max_tokens -= 1
    return llm.decode(generated)


def generate_outfile(functions_definition: List[Dict[str, str]],
                     input: List[Dict[str, str]]) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    vocab_json = parse_infile(llm.get_path_to_vocab_file())
    for prompt in input:
        prompt_string = json.dumps(prompt["prompt"])
        result_string = call_llm(llm, functions_definition, prompt_string, vocab_json)
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
