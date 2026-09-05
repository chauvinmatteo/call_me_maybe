import sys
import json
from pydantic import ValidationError, BaseModel
from typing import Type


def load_data(file_path: str, model_class: Type[BaseModel]) -> list[BaseModel]:

    try:

        with open(file_path) as f:
            data = json.load(f)
            valid_data = []
            for element in data:
                model_checker = model_class(**element)
                valid_data.append(model_checker)

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON.")
        sys.exit(1)

    except ValidationError as e:
        print(f"Error: {e}")
        sys.exit(1)

    return valid_data
