from pydantic import BaseModel
from typing import Dict


class Parameters(BaseModel):
    type: str


class Returns(BaseModel):
    type: str


class FunctionsDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class FunctionsCalling(BaseModel):
    prompt: str
