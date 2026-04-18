import argparse
import json
from sys import exit
from typing import List, Dict
import llm_sdk
import numpy as np


def test_llm(fd: List[Dict[str, str]], input: List[Dict[str, str]]) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    fd_string = json.dumps(fd)
    prompt = (f"Pick a function matching the question 'What is the sum of 2 a"
              f"nd 3?' out of the following: {fd_string} and return the name")
    tokens = llm.encode(prompt)
    # decoded = llm.decode(tokens)
    tokens_list = tokens[0].tolist()
    # print(parse_infile(llm.get_path_to_tokenizer_file()))
    i = 50
    while i:
        logits = llm.get_logits_from_input_ids(tokens_list)
        next_token = np.argmax(logits)
        tokens_list.append(next_token)
        i -= 1
    print(f"Tokens: {tokens_list}\n")
    print(f"String: {llm.decode(tokens_list)}")
    # vocab_file = llm.get_path_to_vocab_file()
    # print(vocab_file)
    # vocab_content = parse_infile(vocab_file)
    # special_tokens = {}
    # for key in vocab_content.keys():
    #     if "endof" in key:
    #         special_tokens[key] = vocab_content[key]
    # print(special_tokens)
    # print(f"Tokens: {tokens}")
    # print(f"String: {decoded}")
    # print(f"Logits: {logits}")
    # print(tokens.shape)


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
    test_llm(functions_definition, input)


if __name__ == "__main__":
    main()
