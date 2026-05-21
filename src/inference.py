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
                   f"{fn_string} out of the prompt '{prompt_string}' "
                   "and only return only a valid JSON object")
    full_prompt_tokens = llm.encode(full_prompt)[0].tolist()
    param_keys = [key for key in function.parameters]
    param_tokens = []
    for key in param_keys:
        param_tokens.append(llm.encode(key)[0].tolist())
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


def get_result_object(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str
) -> str:
    functions_context = get_functions_context(llm,
                                              functions_definition,
                                              prompt_string)
    function_name = get_function_name(llm, functions_context)
    return function_name


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
