import argparse
import json
from sys import exit
from typing import List, Dict, Any
import llm_sdk
import numpy as np


def encode_list(llm: Any, string_list: List[str]) -> List[List[int]]:
    token_list = []
    for string in string_list:
        template_token = llm.encode(string)
        token_list.append(template_token[0].tolist())
    return token_list


def get_param_template(llm: Any, name: str, functions_definition: List[Dict[str, str]]) -> List[List[str]]:
    for definition in functions_definition:
        if definition["name"] == name:
            raw_keys = [key for key in definition["parameters"]]
            types = [definition["parameters"][key]["type"] for key in definition["parameters"]]
    key_strings = []
    for i, key in enumerate(raw_keys):
        if i == 0:
            key_strings.append(f"\"{key}\": ")
        else:
            key_strings.append(f", \"{key}\": ")
    return [encode_list(llm, key_strings), types]


def is_valid_number(next_token_str: str, j: int) -> bool:
    if j == 0 and not (next_token_str.isnumeric() or next_token_str == "-"):
        return False
    if j > 0 and not (next_token_str.isnumeric() or next_token_str == "."):
        return False
    return True

def call_llm(llm: Any, functions_definition: List[Dict[str, str]],
             prompt_string: str) -> str:
    fd_string = json.dumps(functions_definition)
    full_prompt = (f"Pick a function matching the question "
                   f"'{prompt_string}' out of the following: "
                   f"{fd_string} and return only a JSON containing "
                   f"prompt, name and the parameters. If you generate "
                   f"a regular expression, make sure it matches the requested syntax.")
    full_prompt_tokens = llm.encode(full_prompt)
    full_prompt_tokens_list = full_prompt_tokens[0].tolist()
    max_tokens = 100
    state = "START"
    template_list = [f"{{\"prompt\": {prompt_string}, \"name\": \"",
                     ", \"parameters\": {", "}}"]
    template_tokens = encode_list(llm, template_list)
    fd_name_list = [f"{item['name']}\"" for item in functions_definition]
    fd_name_tokens = encode_list(llm, fd_name_list)
    i = 0
    generated = []
    name: List[int] = []
    param_template = None
    in_tokens = True
    j = 0
    while max_tokens:
        logits = llm.get_logits_from_input_ids(full_prompt_tokens_list)
        logits_array = np.array(logits)
        masked = np.full(len(logits), -np.inf)
        if state == "START" and i < len(template_tokens[0]):
            masked[template_tokens[0][i]] = logits_array[template_tokens[0][i]]
            next_token = np.argmax(masked)
            i += 1
        elif state == "NAME":
            for fd_name_token in fd_name_tokens:
                if not name or (len(fd_name_token) > i
                                and fd_name_token[i - 1] == name[i - 1]):
                    masked[fd_name_token[i]] = logits_array[fd_name_token[i]]
            next_token = np.argmax(masked)
            name.append(int(next_token))
            i += 1
        elif state == "PARAM_KEY" and i < len(template_tokens[1]):
            masked[template_tokens[1][i]] = logits_array[template_tokens[1][i]]
            next_token = np.argmax(masked)
            i += 1
        elif state == "PARAM_VALUE" and not param_template:
            param_template = get_param_template(llm, llm.decode(name).replace("\"", ""), functions_definition)
            continue
        elif state == "PARAM_VALUE" and i < len(param_template[0]):
            if in_tokens:
                if j < len(param_template[0][i]):
                    if j == 0 and llm.decode(generated[-1]).endswith(","):
                        j += 1
                        continue
                    masked[param_template[0][i][j]] = logits_array[param_template[0][i][j]]
                    next_token = np.argmax(masked)
                    j += 1
                else:
                    j = 0
                    in_tokens = False
                    next_token = None
            else:
                if param_template[1][i] == "string":
                    if j == 0:
                        masked[llm.encode("\"")] = logits_array[llm.encode("\"")]
                        next_token = np.argmax(masked)
                        j += 1
                    elif j > 0 and "\"" in llm.decode([int(np.argmax(logits_array))]):
                        j = 0
                        i += 1
                        in_tokens = True
                        masked[llm.encode("\"")] = logits_array[llm.encode("\"")]
                        if llm.decode([int(np.argmax(logits_array))]).startswith("\""):
                            next_token = np.argmax(masked)
                        else:
                            next_token = np.argmax(logits_array)
                    else:
                        next_token = np.argmax(logits_array)
                        j += 1
                elif param_template[1][i] == "number":
                    if not is_valid_number(llm.decode([int(np.argmax(logits_array))]), j):
                        j = 0
                        i += 1
                        in_tokens = True
                        next_token = None
                    else:
                        next_token = np.argmax(logits_array)
                        j += 1
        elif state == "END" and i < len(template_tokens[2]):
            masked[template_tokens[2][i]] = logits_array[template_tokens[2][i]]
            next_token = np.argmax(masked)
            i += 1
        else:
            break
        if next_token is not None:
            generated.append(int(next_token))
            full_prompt_tokens_list.append(int(next_token))
        if state == "START" and i >= len(template_tokens[0]):
            state = "NAME"
            i = 0
        elif state == "NAME" and name in fd_name_tokens:
            state = "PARAM_KEY"
            i = 0
        elif state == "PARAM_KEY" and i >= len(template_tokens[1]):
            state = "PARAM_VALUE"
            i = 0
        elif state == "PARAM_VALUE" and i >= len(param_template[0]):
            state = "END"
            i = 0
        elif state == "END" and i >= len(template_tokens[2]):
            break
        try:
            json.loads(llm.decode(generated))
            break
        except Exception:
            pass
        max_tokens -= 1
    return str(llm.decode(generated))


def generate_outfile(functions_definition: List[Dict[str, str]],
                     input: List[Dict[str, str]]) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    for prompt in input:
        prompt_string = json.dumps(prompt["prompt"])
        result_string = call_llm(llm, functions_definition, prompt_string)
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
