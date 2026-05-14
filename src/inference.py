import json
from typing import Optional, Tuple, List, Dict, Any
import llm_sdk
import numpy as np
import os
from src.parsing import FunctionsDefinition, InputPrompt


def encode_list(llm: Any, string_list: List[str]) -> List[List[int]]:
    token_list: List[List[int]] = []
    for string in string_list:
        template_tokens: List[int] = llm.encode(string)[0].tolist()
        token_list.append(template_tokens)
    return token_list


def get_param_template(llm: Any, name: str,
                       functions_definition: List[FunctionsDefinition]
                       ) -> Tuple[List[List[int]], List[str]]:
    for definition in functions_definition:
        if definition.name == name:
            raw_keys: List[str] = [key for key in definition.parameters]
            types: List[str] = [definition.parameters[key]["type"]
                                for key in definition.parameters]
    key_strings: List[str] = []
    for i, key in enumerate(raw_keys):
        if i == 0:
            key_strings.append(f"\"{key}\":"
                               f"{'' if types[i] == 'number' else ' '}")
        else:
            key_strings.append(f", \"{key}\":"
                               f"{'' if types[i] == 'number' else ' '}")
    return (encode_list(llm, key_strings), types)


def get_static_tokens(llm: Any,
                      functions_definition: List[FunctionsDefinition],
                      prompt_string: str
                      ) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    fd_string: str = json.dumps([fd.model_dump() for fd
                                in functions_definition])
    full_prompt: str = ("Pick a function matching the question "
                        f"'{prompt_string}' out of the following: "
                        f"{fd_string} and return only a JSON containing "
                        "prompt, name and the parameters. If you generate "
                        "a regular expression, make sure it matches the "
                        "requested syntax.")
    full_prompt_tokens: List[int] = llm.encode(full_prompt)[0].tolist()
    template_list: List[str] = [f"{{\"prompt\": {prompt_string}, \"name\": \"",
                                ", \"parameters\": {", "}}"]
    template_tokens: List[List[int]] = encode_list(llm, template_list)
    fd_name_list: List[str] = [f"{item.name}\"" for item
                               in functions_definition]
    fd_name_tokens: List[List[int]] = encode_list(llm, fd_name_list)
    return (full_prompt_tokens, template_tokens, fd_name_tokens)


def is_valid_token(in_escape_sequence: bool, next_token_str: str,
                   is_not_last: bool) -> Tuple[bool, bool, bool]:
    if in_escape_sequence:
        token_str: str = "\\" + next_token_str
    else:
        token_str = next_token_str
    i: int = 0
    token_str_len: int = len(token_str)
    in_escape_sequence = False
    while i < token_str_len:
        if in_escape_sequence:
            if token_str[i] not in "\\\"/bfnrt":
                return (False, False, False)
            else:
                in_escape_sequence = False
        else:
            if token_str[i] == "\\":
                in_escape_sequence = True
            elif ((i == token_str_len - 1) and (token_str[i] == "\"")) \
                    or ((is_not_last and i == token_str_len - 2)
                        and (token_str[i:] == "\",")):
                if in_escape_sequence:
                    return (True, False, False)
                else:
                    return (True, False, True)
            elif (i < token_str_len - 1) and (token_str[i] == "\""):
                return (False, False, False)
        i += 1
    return (True, in_escape_sequence, False)


def call_llm(llm: Any, functions_definition: List[FunctionsDefinition],
             prompt_string: str) -> str:
    static_tokens: Tuple[Any, ...] = get_static_tokens(llm,
                                                       functions_definition,
                                                       prompt_string)
    full_prompt_tokens: List[int] = static_tokens[0]
    template_tokens: List[List[int]] = static_tokens[1]
    fd_name_tokens: List[List[int]] = static_tokens[2]
    max_tokens: int = 200
    generated: List[int] = []
    name: List[int] = []
    param_template: Optional[Tuple[List[List[int]], List[str]]] = None
    i: int = 0
    j: int = 0
    state: str = "START"
    template_states: List[str] = ["START", "PARAM_KEY", "END"]
    in_param_template: bool = True
    in_escape_sequence: bool = False
    while max_tokens:
        logits: List[float] = llm.get_logits_from_input_ids(full_prompt_tokens)
        logits_array: np.ndarray[Any, np.dtype[Any]] = np.array(logits)
        masked: np.ndarray[Any, np.dtype[Any]] = np.full(len(logits), -np.inf)
        if state in template_states:
            state_index: int = template_states.index(state)
            token_id: int = template_tokens[state_index][i]
            masked[token_id] = logits_array[token_id]
            next_token: np.intp | None = np.argmax(masked)
            i += 1
        elif state == "NAME":
            for fd_name_token in fd_name_tokens:
                if not name or (len(fd_name_token) > i
                                and fd_name_token[i - 1] == name[i - 1]):
                    masked[fd_name_token[i]] = logits_array[fd_name_token[i]]
            next_token = np.argmax(masked)
            name.append(int(next_token))
            i += 1
        elif state == "PARAM_VALUE" and not param_template:
            name_stripped: str = llm.decode(name).replace("\"", "")
            param_template = get_param_template(llm,
                                                name_stripped,
                                                functions_definition)
            continue
        elif state == "PARAM_VALUE" and param_template:
            next_token_str: str = llm.decode([int(np.argmax(logits_array))])
            if in_param_template:
                if j < len(param_template[0][i]):
                    if j == 0 and llm.decode(generated[-1]).endswith(","):
                        j += 1
                        continue
                    token_id = param_template[0][i][j]
                    masked[token_id] = logits_array[token_id]
                    next_token = np.argmax(masked)
                    j += 1
                else:
                    j = 0
                    in_param_template = False
                    next_token = None
            else:
                if param_template[1][i] == "string":
                    quote_id: List[int] = llm.encode("\"")[0].tolist()
                    esc_quote_id: List[int] = llm.encode("\\\"")[0].tolist()
                    if j == 0:
                        masked[quote_id] = logits_array[quote_id]
                        next_token = np.argmax(masked)
                        j += 1
                    else:
                        is_not_last: bool = i < len(param_template[0]) - 1
                        while True:
                            result: Tuple[bool, bool, bool] = is_valid_token(
                                in_escape_sequence, next_token_str, is_not_last
                                )
                            token_valid, next_in_esc, string_closed = result
                            if not token_valid:
                                logits_array[np.argmax(logits_array)] = -np.inf
                                curr_max: int = int(np.argmax(logits_array))
                                next_token_str = llm.decode([curr_max])
                            else:
                                next_token = np.argmax(logits_array)
                                break
                        if string_closed:
                            j = 0
                            i += 1
                            in_param_template = True
                            in_escape_sequence = False
                        elif max_tokens == 2 and not in_escape_sequence:
                            masked[quote_id] = logits_array[quote_id]
                            next_token = np.argmax(masked)
                        elif max_tokens == 2 and in_escape_sequence:
                            masked[esc_quote_id] = logits_array[esc_quote_id]
                            next_token = np.argmax(masked)
                        else:
                            j += 1
                        in_escape_sequence = next_in_esc
                elif param_template[1][i] == "number":
                    space_minus_id: List[int] = llm.encode(" -")[0].tolist()
                    space_id: List[int] = llm.encode(" ")[0].tolist()
                    dummy_id: List[int] = llm.encode("0")[0].tolist()
                    if j == 0:
                        masked[space_minus_id] = logits_array[space_minus_id]
                        masked[space_id] = logits_array[space_id]
                        next_token = np.argmax(masked)
                        j += 1
                    elif not (next_token_str.isnumeric()
                              or next_token_str == "."):
                        j = 0
                        i += 1
                        in_param_template = True
                        next_token = None
                    elif max_tokens == 2:
                        masked[dummy_id] = logits_array[dummy_id]
                        next_token = np.argmax(masked)
                    else:
                        next_token = np.argmax(logits_array)
                        j += 1
                elif param_template[1][i] == "boolean":
                    bool_list: List[str] = ["true", "false"]
                    bool_tokens: List[List[int]] = encode_list(llm, bool_list)
                    for item in bool_tokens:
                        masked[item] = logits_array[item]
                    next_token = np.argmax(masked)
                    j = 0
                    i += 1
                    in_param_template = True
        if next_token is not None:
            generated.append(int(next_token))
            full_prompt_tokens.append(int(next_token))
        if state == "START" and i >= len(template_tokens[0]):
            state = "NAME"
            i = 0
        elif state == "NAME" and name in fd_name_tokens:
            state = "PARAM_KEY"
            i = 0
        elif state == "PARAM_KEY" and i >= len(template_tokens[1]):
            state = "PARAM_VALUE"
            i = 0
        elif (state == "PARAM_VALUE" and param_template
              and i >= len(param_template[0])) or max_tokens == 2:
            state = "END"
            i = 0
        elif state == "END" and i >= len(template_tokens[2]):
            break
        max_tokens -= 1
    return str(llm.decode(generated))


def generate_outfile(functions_definition: List[FunctionsDefinition],
                     input_prompts: List[InputPrompt], output_path: str
                     ) -> None:
    llm: Any = llm_sdk.Small_LLM_Model()  # type: ignore
    input_len: int = len(input_prompts)
    json_from_file: List[Dict[str, Any]] = []
    for i, item in enumerate(input_prompts, 1):
        print(f"\nProcessing prompt {i}/{input_len}...")
        try:
            prompt_string: str = json.dumps(item.prompt)
            result_string: str = call_llm(llm, functions_definition,
                                          prompt_string)
            result_json: Any = json.loads(result_string)
            print(json.dumps(result_json, indent=2))
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            json_from_file.append(result_json)
            with open(output_path, "w") as f:
                json.dump(json_from_file, f, indent=2)
        except json.JSONDecodeError as e:
            print("Error while generating function call "
                  f"for prompt '{prompt_string}':\n{e}")
        except OSError as e:
            print("Error while writing function call to file "
                  f"for prompt '{prompt_string}':\n{e.strerror}")
