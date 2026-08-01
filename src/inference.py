"""Constrained decoding for function-calling with a small LLM."""

import json
import os
import re
from sys import stderr
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel

import llm_sdk
from src.parsing import FunctionsDefinition, InputPrompt

NUMERIC_REGEX = re.compile(r"^[0-9.\-eE\s]+$")
INTEGER_REGEX = re.compile(r"^[0-9\-]+$")
NUMBER_PREFIX_REGEX = re.compile(r"^-?(0|[1-9][0-9]*)?"
                                 r"(\.[0-9]*)?"
                                 r"([eE][+-]?[0-9]*)?$")
INTEGER_PREFIX_REGEX = re.compile(r"^-?(0|[1-9][0-9]*)?$")


class FunctionsContext(BaseModel):
    """Holds the context for the function selection."""

    prompt_tokens: List[int]
    functions_tokens: List[List[int]]


class ParametersContext(BaseModel):
    """Holds the context for the parameters extraction."""

    prompt_tokens: List[int]
    param_tokens: List[List[int]]
    param_types: List[str]


def load_vocab_mappings(
    llm: Any
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """Parse the vocab file and get mappings for specific tokens.

    Args:
        llm: instance of the llm required to get vocab file.

    Returns:
        Tuple of an inversed vocabulary and allowed tokens for
            constrained decoding.
    """
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_str = {}
    categories: Dict[str, List[int]] = {"number": [],
                                        "integer": [],
                                        "number_end": [],
                                        "boolean": [],
                                        "string_safe": [],
                                        "backslash": [],
                                        "quote": [],
                                        "escape": []}
    for token_str, token_id in vocab.items():
        clean_str = token_str.replace("\u0120", " ")
        id_to_str[token_id] = clean_str
        if (NUMERIC_REGEX.match(clean_str)
                and clean_str.strip() and len(clean_str) == 1):
            categories["number"].append(token_id)
        if (INTEGER_REGEX.match(clean_str)
                and clean_str.strip() and len(clean_str) == 1):
            categories["integer"].append(token_id)
        if clean_str.strip() in [",", "}", "}}"]:
            categories["number_end"].append(token_id)
        if clean_str.strip() in ["true", "false"]:
            categories["boolean"].append(token_id)
        if "\"" not in clean_str and clean_str.isascii() and (
                clean_str.strip() == "\\" or "\\" not in clean_str):
            categories["string_safe"].append(token_id)
        if clean_str.strip() == "\\":
            categories["backslash"].append(token_id)
        if clean_str in ["\"", " \"", "\",", "\"}", "\"}}"]:
            categories["quote"].append(token_id)
        if clean_str in "\"\\/bfnrt" and len(clean_str) == 1:
            categories["escape"].append(token_id)
    return id_to_str, categories


def get_functions_context(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str
) -> FunctionsContext:
    """Build a FunctionsContext instance.

    Args:
        llm: instance of the llm required to get vocab file.
        functions_definition: list of available functions for the llm to pick.
        prompt_string: current prompt telling the llm which function to choose.

    Returns:
        Instance of FunctionsContext to be used in the function selection.
    """
    fd_string = json.dumps([fd.model_dump() for fd
                            in functions_definition])
    full_prompt = ("Pick the function matching the following prompt:"
                   f"\n\n{prompt_string}\n\n out of the following options:\n\n"
                   f"{fd_string}\n\nMake sure to decide based on all available"
                   " informations. Return only the name.\n\n")
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
    """Build a ParametersContext instance.

    Args:
        llm: instance of the llm required to get vocab file.
        functions_definition: list of available functions for the llm to pick.
        prompt_string: current prompt telling the llm which function to choose.
        function_name: name of the currently selected function.

    Returns:
        Instance of ParametersContext to be used in the parameter extraction.
    """
    for fd in functions_definition:
        if fd.name == function_name:
            function = fd
            break
    fn_string = json.dumps(function.model_dump())
    full_prompt = (f"Given the function '{function_name}' with definition "
                   f"'{fn_string}', extract the arguments from the prompt "
                   f"'{prompt_string}'. Always use correct JSON escape"
                   " sequences, where applicable. If regular expressions are "
                   "required, choose the simplest approach. Return only a "
                   "valid JSON object.\n\n")
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
    """Pick the matching function from all available functions.

    Args:
        llm: instance of the llm required to get vocab file.
        functions_context: instance containing the already tokenized prompt and
            tokens for all available functions.

    Returns:
        The selected function name as string.
    """
    i = 0
    max_tokens = len(max(functions_context.functions_tokens, key=len))
    generated: List[int] = []
    while i < max_tokens:
        logits_base = functions_context.prompt_tokens + generated
        logits = np.array(llm.get_logits_from_input_ids(logits_base))
        masked = np.full(len(logits), -np.inf)
        for tokens in functions_context.functions_tokens:
            if len(tokens) > i and tokens[:i] == generated[:i]:
                masked[tokens[i]] = logits[tokens[i]]
        generated.append(int(np.argmax(masked)))
        if generated in functions_context.functions_tokens:
            break
        i += 1
    return str(llm.decode(generated))


def apply_mask(
    mask: List[int],
    logits: np.ndarray[Any, np.dtype[Any]]
) -> np.ndarray[Any, np.dtype[Any]]:
    """Apply the currently selected mask on the logits.

    Args:
        mask: list of all currently allowed tokens.
        logits: the raw array of logits returned by the llm.

    Returns:
        Masked array of logits set to -inf for non-allowed tokens.
    """
    masked = np.full(len(logits), -np.inf)
    masked[mask] = logits[mask]
    return masked


def generate_value(
    llm: Any,
    context_tokens: List[int],
    get_candidates: Callable[[List[int]], List[int]],
    stop_ids: List[int],
    max_tokens: int,
    stop_allowed: Callable[[List[int]], bool]
) -> List[int]:
    """Generate tokens until a stop token or the token limit is reached.

    Args:
        llm: instance of the llm required to get vocab file.
        context_tokens: tokens of prompt and already generated JSON.
        get_candidates: function that returns the available ids for masking.
        stop_ids: token ids that stop the generation.
        max_tokens: token limit to break in case of a generation loop.
        stop_allowed: function that prevents premature stop of generation.

    Returns:
        List of token ids containing the generated value.
    """
    generated_value: List[int] = []
    i = 0
    while i < max_tokens:
        logits = np.array(llm.get_logits_from_input_ids(context_tokens
                                                        + generated_value))
        candidate_ids = get_candidates(generated_value)
        active_stop = stop_ids if stop_allowed(generated_value) else []
        masked_logits = apply_mask(candidate_ids + active_stop, logits)
        next_token = np.argmax(masked_logits)
        if next_token in stop_ids:
            break
        generated_value.append(int(next_token))
        i += 1
    return generated_value


def get_ids(
    categories: Dict[str, List[int]],
    id_to_str: Dict[int, str],
    param_type: str
) -> Tuple[Callable[[List[int]], List[int]],
           List[int], int, Callable[[List[int]], bool]]:
    """Get utilities for the constrained decoding based on the parameter type.

    Args:
        categories: raw tokens for the masking.
        id_to_str: inverted vocab for filtering token ids.
        param_type: current parameter type.

    Returns:
        Tuple containing specific candidates function, stop ids, token limit
            and stop function.
    """

    def number_candidates(generated_value: List[int]) -> List[int]:
        """Return number tokens that keep the value a valid numeric prefix.

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            List of allowed token ids.
        """
        current_str = "".join(id_to_str[t] for t in generated_value)
        return [t for t in categories["number"]
                if NUMBER_PREFIX_REGEX.match(current_str + id_to_str[t])]

    def integer_candidates(generated_value: List[int]) -> List[int]:
        """Return integer tokens that keep the value a valid integer prefix.

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            List of allowed token ids.
        """
        current_str = "".join(id_to_str[t] for t in generated_value)
        return [t for t in categories["integer"]
                if INTEGER_PREFIX_REGEX.match(current_str + id_to_str[t])]

    def string_candidates(generated_value: List[int]) -> List[int]:
        """Return safe string tokens, or escape chars after a backslash.

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            List of allowed token ids.
        """
        if generated_value and generated_value[-1] in categories["backslash"]:
            return categories["escape"]
        return categories["string_safe"]

    def boolean_candidates(generated_value: List[int]) -> List[int]:
        """Return the boolean tokens.

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            List of allowed token ids.
        """
        return categories["boolean"]

    def number_stop_allowed(generated_value: List[int]) -> bool:
        """Block stopping mid-decimal or mid-exponent (e.g. "5." or "5e").

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            True if stopping is currently allowed.
        """
        current_str = "".join(id_to_str[t] for t in generated_value).lower()
        if "e" in current_str:
            after_e = current_str.split("e")[-1].lstrip("+-")
            return len(after_e) > 0 and after_e[-1].isdigit()
        if "." in current_str:
            after_dot = current_str.split(".")[-1]
            return len(after_dot) > 0 and after_dot[-1].isdigit()
        return False

    def always_stop(generated_value: List[int]) -> bool:
        """Allow stopping unconditionally.

        Args:
            generated_value: token ids generated so far for this value.

        Returns:
            True.
        """
        return True

    if param_type == "number":
        return (number_candidates, categories["number_end"],
                30, number_stop_allowed)
    elif param_type == "integer":
        return (integer_candidates, categories["number_end"], 30, always_stop)
    elif param_type == "string":
        return (string_candidates, categories["quote"], 100, always_stop)
    elif param_type == "boolean":
        return (boolean_candidates, [], 1, always_stop)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_result_json(
    llm: Any,
    functions_definition: List[FunctionsDefinition],
    prompt_string: str
) -> str:
    """Generate the full function-call JSON for a single prompt.

    Args:
        llm: instance of the llm required to get vocab file.
        functions_definition: list of available functions for the llm to pick.
        prompt_string: current prompt telling the llm which function to choose.

    Returns:
        The generated function call as a JSON string.
    """
    functions_context = get_functions_context(llm,
                                              functions_definition,
                                              prompt_string)
    function_name = get_function_name(llm, functions_context)
    parameters_context = get_parameters_context(llm,
                                                functions_definition,
                                                prompt_string,
                                                function_name)
    id_to_str, categories = load_vocab_mappings(llm)
    generated = llm.encode(f"{{\"prompt\": {json.dumps(prompt_string)}, "
                           f"\"name\": {json.dumps(function_name)}, "
                           f"\"parameters\": {{")[0].tolist()
    i = 0
    while i < len(parameters_context.param_types):
        generated += parameters_context.param_tokens[i]
        get_candidates, stop_ids, max_tokens, stop_allowed = get_ids(
            categories,
            id_to_str,
            parameters_context.param_types[i]
        )
        if parameters_context.param_types[i] == "string":
            generated += llm.encode("\"")[0].tolist()
        context_tokens = parameters_context.prompt_tokens + generated
        generated += generate_value(
            llm,
            context_tokens,
            get_candidates,
            stop_ids,
            max_tokens,
            stop_allowed
        )
        if parameters_context.param_types[i] == "string":
            generated += llm.encode("\"")[0].tolist()
        if i < len(parameters_context.param_types) - 1:
            generated += llm.encode(",")[0].tolist()
        else:
            generated += llm.encode("}}")[0].tolist()
        i += 1
    return str(llm.decode(generated))


def generate_outfile(
    functions_definition: List[FunctionsDefinition],
    input_prompts: List[InputPrompt],
    output_path: str
) -> None:
    """Run inference for all prompts and write results to a file.

    Args:
        functions_definition: list of available functions for the llm to pick.
        input_prompts: list of prompts to process.
        output_path: path to write the resulting JSON array to.
    """
    llm = llm_sdk.Small_LLM_Model()  # type: ignore
    input_len = len(input_prompts)
    json_from_file = []
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(input_prompts, 1):
        print(f"\nProcessing prompt {i}/{input_len}...")
        try:
            result_string = get_result_json(llm, functions_definition,
                                            item.prompt)
            result_json = json.loads(result_string)
            print(json.dumps(result_json, indent=2))
            json_from_file.append(result_json)
            with open(output_path, "w") as f:
                json.dump(json_from_file, f, indent=2)
        except json.JSONDecodeError as e:
            print("Error while generating function call "
                  f"for prompt '{item.prompt}':\n{e}", file=stderr)
        except OSError as e:
            print("Error while writing function call to file "
                  f"for prompt '{item.prompt}':\n{e.strerror}", file=stderr)
