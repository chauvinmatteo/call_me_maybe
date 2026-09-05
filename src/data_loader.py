import sys
import json
from pydantic import ValidationError, BaseModel
from typing import Type


def load_data(file_path: str, model_class: Type[BaseModel]) -> list[BaseModel]:
    """
    Function to check data in a file, validate it, and inform
    the program user if something is wrong.

    Args:
        file_path (str): Path to the desired file.
        model_class (Type[BaseModel]): Pydantic model used to
        validate the data.

    Returns:
        list[BaseModel]: A list of validated Pydantic objects.

    Raises:
        FileNotFoundError: If the path is unknown (triggers sys.exit).
        json.JSONDecodeError: If the JSON is not valid (triggers sys.exit).
        ValidationError: If the data does not match the model
        (triggers sys.exit).
    """
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
