from pydantic import BaseModel
from typing import Dict


class Parameters(BaseModel):
    """ Used to check the validity of the given parameters."""
    type: str


class Returns(BaseModel):
    """ Used to check the validity of the given returns."""
    type: str


class FunctionsDef(BaseModel):
    """
    Represents the definition of a function available in the AI model.

    Attributes:
        name (str): exact name of function (ex: 'fn_add_numbers').
        description (str): function purpose.
        parameters (Dict[str, Parameters]): dict including needed parameters.
        returns (Returns): needed object as function return.
    """
    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class FunctionsCalling(BaseModel):
    """ Used to check the validity of the given prompt."""
    prompt: str
