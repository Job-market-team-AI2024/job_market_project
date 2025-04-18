from fastapi import FastAPI, HTTPException
from http import HTTPStatus
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from schemas import FitRequest, FitResponse, PredictRequest, PredictResponse, ModelListResponse, SetRequest, SetResponse, ModelInfoRequest, ModelInfoResponse
import logging
from logging.handlers import RotatingFileHandler
from preprocessing import CustomPreprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

### инициализация прилы
app = FastAPI()

### настройка логирования
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler("logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

logger = logging.getLogger("JobMarketAPI")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


### загрузка дефолтной модели (предобученной)
def load_default_model():
    try:
        logger.info("Loading default model...")
        return joblib.load('jobmarket_model.pkl')
    except FileNotFoundError:
        logger.error("Default model file not found.")
        raise RuntimeError("Default model file not found.")
    except Exception as e:
        logger.error(f"Error loading default model: {str(e)}")
        raise RuntimeError(f"Error loading default model: {str(e)}")


default_model = load_default_model()

### хранилище для обученных моделей (и активной модели)
models: Dict[str, Any] = {'default_model': default_model}
active_models: Dict[str, Any] = {}


### пытались соблюсти условие про отмену обучения через 10 секунд с помощью многопроцессности
### однако, не удалось решить проблему с синхронизацией обновления хранилища моделей через разные процессы
### (так как у каждого процесса свое адресное пространство, память и тд)
### пытались делать через менеджер процессов и/или запись обученных моделей на диск, но не вышло
### поэтому поставили простой, но рабочий вариант
@app.post("/fit", response_model=FitResponse)
async def fit_model(request: FitRequest):
    try:
        logger.info(f"Received fit request for model {request.config.model_id}.")
        df = pd.DataFrame([vacancy.dict() for vacancy in request.data])

        for col in ['salary_from', 'salary_to']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1, skipna=True)
        df = df[df['salary'].notna()]

        df = df[df['salary'] > 10000]
        df['log_salary'] = np.log(df['salary'])

        X = df.drop(['salary_from', 'salary_to', 'salary', 'log_salary'],
                    axis=1,
                    errors="ignore")
        y = df['log_salary']

        cols_to_get_from_name = ['field', 'role', 'grade']
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        ### трансформация колонок
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
        ])
        column_trans = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )

        ### пайплайн с кастомным предпроцессингом
        pipe = Pipeline(steps=[
            ('preprocessor', CustomPreprocessing(cols_to_get_from_name=cols_to_get_from_name)),
            ('column_transformer', column_trans),
            ('regressor', LinearRegression(**request.config.hyperparameters))
        ])
        pipe.fit(X, y)
        models[request.config.model_id] = pipe
        logger.info(f"Model {request.config.model_id} trained and saved.")
        return FitResponse(message=f"Model {request.config.model_id} training completed.")
    except Exception as e:
        logger.error(f"Error training model {request.config.model_id}: {str(e)}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Model training failed.")

def get_pipeline_steps(model):
    steps = []
    if isinstance(model, Pipeline):
        for name, step in model.steps:
            try:
                steps.append(name)  # Append only step names
            except Exception as e:
                steps.append(f"{name} (Error: {e})")
    return steps

### кривые обучения
@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info(request: ModelInfoRequest):
    logger.info(f"Getting models '{request.model_id}' info.")
    model_id = request.model_id

    try:
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' doesn't exist.")

        model = models[model_id]
        df = pd.DataFrame(request.data)

        ### data
        df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1, skipna=True).fillna(0)
        df = df[df['salary'] > 10000]
        X = df.drop(["salary_from", "salary_to", "salary", "log_salary"], axis=1, errors="ignore")
        y = np.log(df['salary'])

        model_steps = [step_name for step_name in get_pipeline_steps(model)]

        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model.named_steps["regressor"],
                model.named_steps["column_transformer"].transform(X),
                y,
                cv=5,
                train_sizes=np.linspace(0.1, 1.0, 6),
                n_jobs=-1
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating learning curve: {str(e)}")

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        ### coefficients and intercept
        regressor = model.named_steps['regressor']
        coefficients = getattr(regressor, 'coef_', None)
        intercept = getattr(regressor, 'intercept_', None)

        return ModelInfoResponse(
            model_info={
                "model_id": model_id,
                "model_steps": model_steps,
                "coefficients": coefficients.tolist() if coefficients is not None else [],
                "intercept": intercept if intercept is not None else 0,
                "learning_curve": {
                    "train_sizes": train_sizes.tolist(),
                    "train_mean": train_mean.tolist(),
                    "train_std": train_std.tolist(),
                    "test_mean": test_mean.tolist(),
                    "test_std": test_std.tolist(),
                }
            }
        )
    except Exception as e:
        logger.error(f"Error getting model info for {model_id}: {str(e)}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Error generating model info: {str(e)}")


### предикт по активной модели
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model_id = request.model_id
    try:
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' doesn't exist.")
        if model_id not in active_models:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_id}' is not active.")
        model = models[model_id]
        X = pd.DataFrame([vacancy.dict() for vacancy in request.data])
        predictions = np.round(np.exp(
            model.predict(X))).tolist()  ### так как обучались на логзарплатах, то возвращаем экспоненту и округляем
        return PredictResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Prediction failed.")


### возврат списка обученных моделей в хранилище с их типом и статусом активности
@app.get("/models", response_model=ModelListResponse)
async def list_models():
    logger.info("Fetching model list.")
    return ModelListResponse(models=[
        {"model_id": model_id,
         "type": type(models[model_id].named_steps['regressor']).__name__,
         "status": "active" if model_id in active_models else "inactive"}
        for model_id, model in models.items()
    ])


### установка активной модели
@app.post("/set", response_model=SetResponse)
async def set_active_model(request: SetRequest):
    logger.info(f"Setting model '{request.model_id}' as active.")
    model_id = request.model_id
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' doesn't exist.")
    if model_id in active_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' is already active")
    active_models.clear()  ### только одна активная модель
    active_models[model_id] = models[model_id]
    logger.info(f"Model '{model_id}' set as active.")
    return SetResponse(message=f"Model '{model_id}' is now active.")
