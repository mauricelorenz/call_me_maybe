import json
from typing import Optional, Tuple, List, Dict, Any
import llm_sdk
import numpy as np
import os
from src.parsing import FunctionsDefinition, InputPrompt
from pydantic import BaseModel, Field, ValidationError


class State(BaseModel):
    main_states: List[str] = Field(default_factory=lambda:
                                   ["START", "PROMPT_KEY",
                                   "PROMPT_VALUE", "NAME_KEY",
                                   "NAME_VALUE", "PARAM_KEY",
                                   "PARAM_VALUE", "END"])
    param_states: List[str] = Field(default_factory=list)
    curr_main_state: int = 0
    curr_param_state: int = 0
    curr_token_index: int = 0
    in_string: bool = False
    in_escape: bool = False

    def get_state(self) -> Tuple[str, str]:
        return (self.main_states[self.curr_main_state],
                self.param_states[self.curr_param_state]
                if self.param_states else "")

    def next_state(self, choice: str) -> None:
        if choice == "main":
            self.curr_token_index = 0
            self.curr_main_state += 1
        elif choice == "param":
            self.curr_token_index = 0
            self.curr_param_state += 1
        elif choice == "token":
            self.curr_token_index += 1
        else:
            raise ValueError(f"Unknown choice: {choice}")


class GenerationContext(BaseModel):
    full_prompt_tokens: List[int]
    structure_tokens: Dict[str, List[int]]
    functions_tokens: List[List[int]]
    param_tokens: List[List[int]] = Field(default_factory=list)
    param_types: List[str] = Field(default_factory=list)


def get_generation_context(llm: Any, functions_definition: List[FunctionsDefinition], prompt_string: str) -> GenerationContext:
    fd_string = json.dumps([fd.model_dump() for fd
                                in functions_definition])
    full_prompt = ("Pick a function matching the question "
                        f"'{prompt_string}' out of the following: "
                        f"{fd_string} and return only a JSON containing "
                        "prompt, name and the parameters. If you generate "
                        "a regular expression, make sure it matches the "
                        "requested syntax.")
    full_prompt_tokens = llm.encode(full_prompt)[0].tolist()
    structure_dict = {"START": "{",
                      "PROMPT_KEY": "\"prompt\":",
                      "PROMPT_VALUE": prompt_string,
                      "NAME_KEY": "\"name\":",
                      "PARAM_KEY": "\"parameters\":",
                      "END": "}"}
    structure_tokens = {}
    for item in structure_dict:
        structure_tokens[item] = llm.encode(structure_dict[item])[0].tolist()
    functions_tokens = []
    for item in functions_definition:
        functions_tokens.append(llm.encode(item.name)[0].tolist())
    return GenerationContext(full_prompt_tokens=full_prompt_tokens, structure_tokens=structure_tokens, functions_tokens=functions_tokens)


def get_next_token(state: State, logits: np.ndarray, generation_context: GenerationContext) -> int:
    main_state, param_state = state.get_state()
    if main_state == "START":
        return int(np.argmax(logits))


def call_llm(llm: Any, functions_definition: List[FunctionsDefinition],
             prompt_string: str) -> str:
    state = State()
    max_tokens = 100
    generation_context = get_generation_context(llm, functions_definition, prompt_string)
    generated = []
    while max_tokens:
        logits = np.array(llm.get_logits_from_input_ids(generation_context.full_prompt_tokens + generated))
        token = get_next_token(state, logits, generation_context)
        generated.append(token)
        max_tokens -= 1
    return llm.decode(generated)


def generate_outfile(functions_definition: List[FunctionsDefinition],
                     input_prompts: List[InputPrompt], output_path: str
                     ) -> None:
    llm = llm_sdk.Small_LLM_Model()
    input_len = len(input_prompts)
    json_from_file = []
    for i, item in enumerate(input_prompts, 1):
        print(f"\nProcessing prompt {i}/{input_len}...")
        try:
            prompt_string = json.dumps(item.prompt)
            result_string = call_llm(llm, functions_definition,
                                          prompt_string)
            result_json = json.loads(result_string)
            print(json.dumps(result_json, indent=2))
            if not os.path.exists(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            json_from_file.append(result_json)
            with open(output_path, "w") as f:
                json.dump(json_from_file, f, indent=2)
        except json.JSONDecodeError as e:
            print(result_string)  ######################################################################## DEBUG
            print("Error while generating function call "
                  f"for prompt '{prompt_string}':\n{e}")
        except OSError as e:
            print("Error while writing function call to file "
                  f"for prompt '{prompt_string}':\n{e.strerror}")
