import json
from sys import exit
from typing import List, Dict, Type, TypeVar, Any
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class FunctionsDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class InputPrompt(BaseModel):
    prompt: str


def parse_infile(path: str, model: Type[T]) -> List[T]:
    try:
        with open(path) as f:
            data: Any = json.load(f)
            return [model(**item) for item in data]
    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError, ValidationError) as e:
        if isinstance(e, (FileNotFoundError, PermissionError)):
            print(f"Error while parsing '{path}': {e.strerror}")
        elif isinstance(e, ValidationError):
            print(f"Error while parsing '{path}':\n{e.errors()[0]['msg']}: "
                  f"'{e.errors()[0]['loc'][0]}'")
        else:
            print(f"Error while parsing '{path}':\n{e}")
        exit(1)
