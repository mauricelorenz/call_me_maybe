import json
from sys import exit
from typing import List, Dict, Type
from pydantic import BaseModel, ValidationError


class FunctionsDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class InputPrompt(BaseModel):
    prompt: str


def parse_infile(path: str, model: Type[BaseModel]) -> List[BaseModel]:
    try:
        with open(path) as f:
            data = json.load(f)
            return [model(**item) for item in data]
    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError, ValidationError) as e:
        if isinstance(e, (FileNotFoundError, PermissionError)):
            print(f"Error while parsing '{path}': {e.strerror}")
        elif isinstance(e, ValidationError):
            err = e.errors()[0]
            print(f"Error while parsing '{path}':\n{err['msg']}: "
                  f"'{err['loc'][0]}'")
        else:
            print(f"Error while parsing '{path}':\n{e}")
        exit(1)
