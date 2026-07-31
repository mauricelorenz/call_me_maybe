"""Input file parsing for function-calling with a small LLM."""

import json
from sys import exit, stderr
from typing import List, Dict, Type, TypeVar, Any
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class FunctionsDefinition(BaseModel):
    """Schema for a single callable function definition."""

    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class InputPrompt(BaseModel):
    """Schema for a single user prompt to process."""

    prompt: str


def parse_infile(path: str, model: Type[T]) -> List[T]:
    """Load and validate a JSON file as a list of the given model.

    Args:
        path: path to the JSON file to parse.
        model: pydantic model each list entry is validated against.

    Returns:
        List of validated model instances.
    """
    try:
        with open(path) as f:
            data: Any = json.load(f)
            return [model(**item) for item in data]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error while parsing '{path}': {e.strerror}", file=stderr)
        exit(1)
    except ValidationError as e:
        print(f"Error while parsing '{path}':\n{e.errors()[0]['msg']}: "
              f"'{e.errors()[0]['loc'][0]}'", file=stderr)
        exit(1)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error while parsing '{path}':\n{e}", file=stderr)
        exit(1)
