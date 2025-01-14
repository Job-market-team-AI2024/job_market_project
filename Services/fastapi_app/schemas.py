import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any

class Vacancy_for_fit(BaseModel):
    name: str
    area_name: str
    employer_name: str
    schedule: str
    employment: str
    experience: str
    # professional_roles_name: str
    # count_key_skills: int
    salary_from: Union[int, float] = None
    salary_to: Union[int, float] = None

class Vacancy_for_predict(BaseModel):
    name: str
    area_name: str
    employer_name: str
    schedule: str
    employment: str
    experience: str

class FitConfig(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    # description: Optional[str] = None
    # ml_model_type: str
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters for the model")

class FitRequest(BaseModel):
    data: List[Vacancy_for_fit] = Field(..., description="Training data for the model")
    config: FitConfig = Field(..., description="Configuration for the model training")

class FitResponse(BaseModel):
    message: str = Field(..., description="Message confirming the training status of the model")

###
class PredictRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to use for prediction")
    data: List[Vacancy_for_predict] = Field(..., description="Data for which predictions are required")

class PredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="Predictions made by the model")

###
class ModelListResponse(BaseModel):
    models: List[Dict[str, str]] = Field(..., description="List of available models")

### set
class SetRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to set as active")
class SetResponse(BaseModel):
    message: str = Field(..., description="Message confirming the model is set as active")

### model_info
class ModelInfoRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to set as active")
    data: Any = Field(..., description="Custom df for analysis")
class ModelInfoResponse(BaseModel):
    model_info: Dict[str, Any] = Field(..., description="List of available models")

