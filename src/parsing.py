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
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error while parsing '{path}': {e.strerror}")
        exit(1)
    except ValidationError as e:
        print(f"Error while parsing '{path}':\n{e.errors()[0]['msg']}: "
              f"'{e.errors()[0]['loc'][0]}'")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error while parsing '{path}':\n{e}")
        exit(1)
