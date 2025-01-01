import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from http import HTTPStatus
from typing import Dict, List, Union, Any
import joblib
from preprocessing import preprocess_data
from schemas import FitRequest, FitResponse, PredictRequest, PredictionResponse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

router = FastAPI()

default_model = joblib.load('jobmarket_model.pkl')

models = {'default_model': default_model}


@router.post("/fit", response_model=FitResponse)
async def fit_model(request: FitRequest):
    X, y = preprocess_data(request.data)
    category_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', category_transformer, X.select_dtypes(include=['object']).columns),
            ('num', numeric_transformer, X.select_dtypes(include=['number']).columns)
        ]
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression(**request.config.hyperparameters))
    ]).fit(X, y)
    model_id = request.config.id
    models[model_id] = model
    return FitResponse(message=f'Model {model_id} trained and saved')


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    model_id = request.model_id
    model = models[model_id]
    X, y = preprocess_data(request.data)
    predictions = model.predict(X).tolist()
    return PredictionResponse(predictions=predictions)
