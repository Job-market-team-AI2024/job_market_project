import streamlit as st
import requests
import plotly.graph_objects as go
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import json
import logging
from logging.handlers import RotatingFileHandler

# Логирование
LOG_DIR = './logs'
LOG_FILE = f'{LOG_DIR}/streamlit.log'
handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Base URL
API_BASE_URL = 'http://localhost:8000'

st.title('Вакансии с порталов для поиска работы')

st.write('''В современном мире поиск работы и привлечение квалифицированных специалистов являются важными 
задачами как для соискателей, так и для работодателей. На порталах трудоустройства, таких как HeadHunter, 
часто встречаются вакансии, в которых отсутствует информация о заработной плате. Это затрудняет оценку 
привлекательности предложений и может привести к неэффективным решениям.
Данный сервис предоставляет аналитический отчёт об объявлдениях о вакансиях на порталах трудоустройства
и предсказывает заработную плату.''')

uploaded_file = st.file_uploader('Upload your dataset')

if uploaded_file: 
    uploaded_data = pd.read_csv(uploaded_file)
    st.dataframe(uploaded_data.head())
    logger.info('Dataset uploaded successfully.')
    
    menu = st.sidebar.selectbox('Menu', ['EDA', 'Train Model & Learning Curves', 'Inference'])
    
    if menu == 'Train Model & Learning Curves':
        st.header('Train a Model with Hyperparameters')
        
        if 'uploaded_data' not in locals():
            st.warning('Please upload a dataset first!')
        else:
            st.sidebar.header('Select Hyperparameter Values')
            learning_rate = st.sidebar.slider('Learning Rate', 0.001, 0.1, 0.01, step=0.001)
            n_estimators = st.sidebar.slider('Number of Estimators', 50, 500, 100, step=10)
            
            model_id = st.text_input('Model ID', value='new_model')
            if st.button('Train Model'):
                try:
                    hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
                    response = requests.post(
                        f'{API_BASE_URL}/fit',
                        json={
                            'data': uploaded_data.to_dict(),
                            'config': {'id': model_id, 'hyperparameters': hyperparams},
                        },
                    )
                    if response.status_code == 200:
                        st.success(response.json().get('message'))
                        logger.info(f'Model {model_id} trained successfully with hyperparameters: {hyperparams}.')
                    else:
                        st.error(f'Error: {response.json().get('detail')}')
                        logger.error(f'Training error: {response.json().get('detail')}')
                except Exception as e:
                    st.error(f'Error during training: {e}')
                    logger.error(f'Training error: {e}')
            
            if st.button('Show Learning Curves'):
                try:
                    response = requests.get(f'{API_BASE_URL}/get-learning-curves/{model_id}')
                    if response.status_code == 200:
                        learning_curves = response.json()
                        train_scores = learning_curves['train_scores']
                        val_scores = learning_curves['val_scores']
                        epochs = range(1, len(train_scores) + 1)
    
                        plt.figure(figsize=(10, 6))
                        plt.plot(epochs, train_scores, label='Training Score', marker='o')
                        plt.plot(epochs, val_scores, label='Validation Score', marker='x')
                        plt.title(f'Learning Curves for Model {model_id}')
                        plt.xlabel('Epochs')
                        plt.ylabel('Score')
                        plt.legend()
                        st.pyplot(plt)
                        logger.info(f'Learning curves displayed for model {model_id}.')
                    else:
                        st.error(f'Error fetching learning curves: {response.json().get('detail')}')
                        logger.error(f'Learning curve error: {response.json().get('detail')}')
                except Exception as e:
                    st.error(f'Error during learning curve display: {e}')
                    logger.error(f'Learning curve display error: {e}')
    
    if menu == 'Inference':
        st.header('Model Inference')
        
        model_id = st.text_input('Enter Model ID for Inference', value='default_model')
        inference_file = st.file_uploader('Upload CSV file for inference', type=['csv'])
        
        if inference_file is not None:
            inference_data = pd.read_csv(inference_file)
            st.dataframe(inference_data.head())
            
            if st.button('Run Inference'):
                try:
                    response = requests.post(
                        f'{API_BASE_URL}/predict',
                        json={'data': inference_data.to_dict(), 'model_id': model_id},
                    )
                    if response.status_code == 200:
                        predictions = response.json().get('predictions', [])
                        st.success('Inference completed successfully!')
                        st.write('Predictions:')
                        st.write(predictions)
                        logger.info(f'Inference successful for model {model_id}. Predictions: {predictions}')
                    else:
                        st.error(f'Error: {response.json().get('detail')}')
                        logger.error(f'Inference error: {response.json().get('detail')}')
                except Exception as e:
                    st.error(f'Error during inference: {e}')
                    logger.error(f'Inference error: {e}')
