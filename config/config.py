from pydantic import BaseModel, field_validator
from typing import Optional


class TrainPayLoad(BaseModel):
    source:str
    train:bool
    filename:Optional[str]

class TestPayLoad(BaseModel):
    source:str
    predict:bool
    modelpath:str

