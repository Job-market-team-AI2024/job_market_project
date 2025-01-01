from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
from enum import Enum


class Vacancy(BaseModel):
    name: str
    area_name: str
    schedule: str
    employment: str
    experience: str
    professional_roles_name: str
    count_key_skills: int
    salary_from: Optional[Union[int, float]] = None
    salary_to: Optional[Union[int, float]] = None


class Vacancies(BaseModel):
    data: List[Vacancy]


class ModelType(str, Enum):
    linear = "linear"


class ModelConfig(BaseModel):
    id: str
    description: Optional[str] = None
    ml_model_type: ModelType
    hyperparameters: Dict[str, Any]


class PredictRequest(BaseModel):
    model_id: str
    data: Vacancies


class PredictionResponse(BaseModel):
    predictions: List[float]


class ModelListResponse(BaseModel):
    models: List[ModelConfig]
