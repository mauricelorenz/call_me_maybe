import json
from typing import List, Any
import llm_sdk
import numpy as np
import os
from src.parsing import FunctionsDefinition, InputPrompt
from pydantic import BaseModel


class FunctionsContext(BaseModel):
    prompt_tokens: List[int]
    functions_tokens: List[List[int]]


class ParametersContext(BaseModel):
    prompt_tokens: List[int]
    param_tokens: List[List[int]]
    param_types: List[str]


class State(BaseModel):
    param_index: int = 0
    token_index: int = 0
    start: bool = True
    in_key: bool = False
    in_value: bool = False
    value_done: bool = False
    end: bool = False
    done: bool = False
    in_string: bool = False
    in_escape: bool = False


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
    full_prompt = ("Extract the parameters matching the definition in "
                   f"{fn_string} out of the prompt '{prompt_string}'. "
                   "Only return a valid JSON object.")
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


def get_next_state(parameters_context: ParametersContext, state: State) -> None:
    if state.start:
        state.start = False
        state.in_key = True
    elif state.in_key and state.token_index == len(parameters_context.param_tokens[state.param_index]):
        state.token_index = 0
        state.in_key = False
        state.in_value = True
    elif state.in_value and state.value_done and state.param_index == len(parameters_context.param_tokens) - 1:
        state.value_done = False
        state.in_value = False
        state.end = True
    elif state.in_value and state.value_done:
        state.param_index += 1
        state.token_index = 0
        state.value_done = False
        state.in_value = False
        state.in_key = True
    elif state.end:
        state.end = False
        state.done = True


def enforce_tokens(llm: Any, to_enforce: List[str], logits: np.ndarray, masked: np.ndarray) -> int:
    for raw_token in to_enforce:
        token = llm.encode(raw_token)[0].tolist()
        masked[token] = logits[token]
    return np.argmax(masked)


def validate_token(token_str: str, state: State) -> bool:
    if state.in_escape:
        token_str = "\\" + token_str
    state.in_escape = False
    i = 0
    token_str_len = len(token_str)
    while i < token_str_len:
        if state.in_escape:
            if token_str[i] in "\\\"/bfnrt":
                state.in_escape = False
            else:
                return False
        elif token_str[i] == "\\":
            state.in_escape = True
        elif token_str[i] == "\"" and i < token_str_len - 1:
            return False
        i += 1
    return True


def get_next_token(llm: Any, parameters_context: ParametersContext, state: State, logits: np.ndarray) -> int | None:
    masked = np.full(len(logits), -np.inf)
    next_token = None
    if state.start:
        token = llm.encode("{")
        masked[token] = logits[token]
        next_token = np.argmax(masked)
    elif state.in_key:
        token = parameters_context.param_tokens[state.param_index][state.token_index]
        masked[token] = logits[token]
        next_token = np.argmax(masked)
        state.token_index += 1
    elif state.in_value:
        probable_token_str = llm.decode([int(np.argmax(logits))])
        if parameters_context.param_types[state.param_index] == "number":
            if state.token_index == 0:
                next_token = enforce_tokens(llm, [" ", "-", " -"], logits, masked)
            elif probable_token_str.isnumeric() or probable_token_str == ".":
                next_token = np.argmax(logits)
            else:
                if state.param_index < len(parameters_context.param_tokens) - 1:
                    next_token = enforce_tokens(llm, [",", ", ", " ,", " , "], logits, masked)
                state.value_done = True
        elif parameters_context.param_types[state.param_index] == "string":
            if state.token_index == 0:
                next_token = enforce_tokens(llm, ["\"", " \""], logits, masked)
                state.in_string = True
            elif state.in_string:
                while True:
                    # print(np.argmax(logits),":", llm.decode([int(np.argmax(logits))]))  # ############################## DEBUG
                    is_valid_token = validate_token(llm.decode([int(np.argmax(logits))]), state)
                    # print(is_valid_token)  # ############################## DEBUG
                    if is_valid_token:
                        next_token = np.argmax(logits)
                        break
                    logits[np.argmax(logits)] = -np.inf
                if llm.decode([int(np.argmax(logits))]).endswith("\"") and not state.in_escape:
                    state.in_string = False
            else:
                if state.param_index < len(parameters_context.param_tokens) - 1:
                    next_token = enforce_tokens(llm, [",", ", ", " ,", " , "], logits, masked)
                state.value_done = True
        elif parameters_context.param_types[state.param_index] == "boolean":
            if state.token_index == 0:
                bool_tokens = [llm.encode(item) for item in ["true", "false"]]
                for token in bool_tokens:
                    masked[token] = logits[token]
                next_token = np.argmax(masked)
            else:
                if state.param_index < len(parameters_context.param_tokens) - 1:
                    next_token = enforce_tokens(llm, [",", ", ", " ,", " , "], logits, masked)
                state.value_done = True
        state.token_index += 1
    elif state.end:
        token = llm.encode("}")
        masked[token] = logits[token]
        next_token = np.argmax(masked)
    get_next_state(parameters_context, state)
    return next_token


def get_parameters(llm: Any, parameters_context: ParametersContext) -> str:
    state = State()
    i = 0
    max_tokens = 100
    generated: List[int] = []
    while i < max_tokens:
        logits_base = parameters_context.prompt_tokens + generated
        logits = np.array(llm.get_logits_from_input_ids(logits_base))
        token = get_next_token(llm, parameters_context, state, logits)
        if token is not None:
            generated.append(token)
        if state.done:
            break
        i += 1
    return str(llm.decode(generated))


def get_result_object(
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
    parameters = get_parameters(llm, parameters_context)
    return (function_name, parameters)


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
            prompt_string = json.dumps(item.prompt)
            result_string = get_result_object(llm, functions_definition,
                                              prompt_string)
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
