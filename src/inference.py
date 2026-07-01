import json
from typing import List, Any, Tuple, Dict
import llm_sdk
import numpy as np
import os
from src.parsing import FunctionsDefinition, InputPrompt
from pydantic import BaseModel
import re


NUMERIC_REGEX = re.compile(r"^[0-9.\-eE\s]+$")


class FunctionsContext(BaseModel):
    prompt_tokens: List[int]
    functions_tokens: List[List[int]]


class ParametersContext(BaseModel):
    prompt_tokens: List[int]
    param_tokens: List[List[int]]
    param_types: List[str]


def load_vocab_mappings(
    llm: Any
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_str = {}
    categories: Dict[str, List[int]] = {"number": [],
                                        "integer": [],
                                        "boolean": [],
                                        "string_safe": [],
                                        "escape": []}
    for token_str, token_id in vocab.items():
        clean_str = token_str.replace("\u0120", " ")
        id_to_str[token_id] = clean_str
        if NUMERIC_REGEX.match(clean_str) and clean_str.strip():
            categories["number"].append(token_id)
        if clean_str.isnumeric():
            categories["integer"].append(token_id)
        if clean_str.strip().lower() in ["true", "false"]:
            categories["boolean"].append(token_id)
        if "\"" not in clean_str:
            categories["string_safe"].append(token_id)
        if clean_str in "\"\\/bfnrt":
            categories["escape"].append(token_id)
    return id_to_str, categories


def get_functions_context(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str
) -> FunctionsContext:
    fd_string = json.dumps([fd.model_dump() for fd
                            in functions_definition])
    full_prompt = ("Pick the function matching the prompt "
                   f"'{prompt_string}' out of the following:\n\n"
                   f"{fd_string}\n\nReturn only the name.")
    full_prompt_tokens = llm.encode(full_prompt)[0].tolist()
    functions_tokens = []
    for item in functions_definition:
        functions_tokens.append(llm.encode(item.name)[0].tolist())
    return FunctionsContext(prompt_tokens=full_prompt_tokens,
                            functions_tokens=functions_tokens)


def get_parameters_context(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str,
    function_name: str
) -> ParametersContext:
    for fd in functions_definition:
        if fd.name == function_name:
            function = fd
            break
    fn_string = json.dumps(function.model_dump())
    full_prompt = (f"Given the function '{function_name}' with definition "
                   f"'{fn_string}', extract the arguments from the prompt "
                   f"'{prompt_string}'. Return only a valid JSON object with "
                   "the parameter values.")
    full_prompt_tokens = llm.encode(full_prompt)[0].tolist()
    param_keys = [key for key in function.parameters]
    param_tokens = []
    for key in param_keys:
        param_tokens.append(llm.encode(f"\"{key}\":")[0].tolist())
    param_types = [function.parameters[key]["type"]
                   for key in function.parameters]
    return ParametersContext(prompt_tokens=full_prompt_tokens,
                             param_tokens=param_tokens,
                             param_types=param_types)


def get_function_name(llm: Any, functions_context: FunctionsContext) -> str:
    i = 0
    max_tokens = len(max(functions_context.functions_tokens, key=len))
    generated: List[int] = []
    while i < max_tokens:
        logits_base = functions_context.prompt_tokens + generated
        logits = np.array(llm.get_logits_from_input_ids(logits_base))
        masked = np.full(len(logits), -np.inf)
        for tokens in functions_context.functions_tokens:
            if not generated or (len(tokens) > i
                                 and tokens[i - 1] == generated[i - 1]):
                masked[tokens[i]] = logits[tokens[i]]
        generated.append(int(np.argmax(masked)))
        if generated in functions_context.functions_tokens:
            break
        i += 1
    return str(llm.decode(generated))


def get_mask(state: str, categories: Dict[str, List[int]]) -> List[int]:
    if state == "IN_NUMBER":
        return categories["number"]
    elif state == "IN_INTEGER":
        return categories["integer"]
    elif state == "IN_BOOLEAN":
        return categories["boolean"]
    elif state == "IN_STRING":
        return categories["string_safe"]
    elif state == "IN_ESCAPE":
        return categories["escape"]
    else:
        raise ValueError(f"Unknown state: {state}")


def apply_mask(
    mask: List[int],
    logits: np.ndarray[Any, np.dtype[Any]]
) -> np.ndarray[Any, np.dtype[Any]]:
    masked = np.full(len(logits), -np.inf)
    masked[mask] = logits[mask]
    return masked


def get_result_json(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str
) -> str:
    functions_context = get_functions_context(llm,
                                              functions_definition,
                                              prompt_string)
    function_name = get_function_name(llm, functions_context)
    parameters_context = get_parameters_context(llm,
                                                functions_definition,
                                                prompt_string,
                                                function_name)
    id_to_str, categories = load_vocab_mappings(llm)
    state = "START"
    i = 0
    max_tokens = 100
    generated: List[np.signedinteger] = []
    while i < max_tokens:
        logits_base = parameters_context.prompt_tokens + generated
        logits = np.array(llm.get_logits_from_input_ids(logits_base))
        mask = get_mask(state, categories)
        masked_logits = apply_mask(mask, logits)  # noqa: F841
        # TODO: add sampling, fix literals and state transition
        if state == "DONE":
            break
        i += 1
    return str(llm.decode(generated))


def generate_outfile(
    functions_definition: List[FunctionsDefinition],
    input_prompts: List[InputPrompt],
    output_path: str
) -> None:
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    input_len = len(input_prompts)
    json_from_file = []
    for i, item in enumerate(input_prompts, 1):
        print(f"\nProcessing prompt {i}/{input_len}...")
        try:
            result_string = get_result_json(llm, functions_definition,
                                              item.prompt)
            result_json = json.loads(result_string)
            print(json.dumps(result_json, indent=2))
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            json_from_file.append(result_json)
            with open(output_path, "w") as f:
                json.dump(json_from_file, f, indent=2)
        # except json.JSONDecodeError as e:
        #     print("Error while generating function call "
        #           f"for prompt '{prompt_string}':\n{e}")
        # except OSError as e:
        #     print("Error while writing function call to file "
        #           f"for prompt '{prompt_string}':\n{e.strerror}")
        except Exception:
            print(item)
            print(result_string)  # ##################################### DEBUG
