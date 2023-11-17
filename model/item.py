from pydantic import BaseModel, Field
from typing import Union
from fastapi import Query

class DataInput(BaseModel):
    firstMajor: str
    applyGrade: str = Query(regex="^\d-\d$") # 2-1 format
    applyMajor: str
    applySemester: str = Query(regex="^\d{4}-\d$") # 2023-1 format
    applyGPA: float = Field(ge=0, le=4.5) # 0 ~ 4.5

class PredictOutput(BaseModel):
    result: int # 1이면 합격, 0이면 불합격 -> 지금은 2개의 클래스로 