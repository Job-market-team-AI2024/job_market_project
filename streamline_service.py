import streamlit as st
import requests
import plotly.graph_objects as go

st.title('title')

# 1. Загрузка датасета
uploaded_file = st.file_uploader('Upload your dataset')
if uploaded_file:
    response = requests.post('http://fastapi:8000/upload-dataset/', files={'file': uploaded_file})
    st.success(response.json().get('message'))

    # 2. Настройка гиперпараметров
    st.sidebar.header('Hyperparameters')
    learning_rate = st.sidebar.slider('Learning Rate', 0.001, 0.1, 0.01)
    n_estimators = st.sidebar.slider('Number of Estimators', 50, 500, 100)
    
    if st.button('Train Model'):
        hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
        response = requests.post('http://fastapi:8000/fit/', json=hyperparams)
        st.success(response.json().get('message'))
    
    # 3. Просмотр кривых обучения
    st.subheader('Training Curves')
    selected_models = st.multiselect('Select models to compare', options=['model_1', 'model_2', 'model_3'])
    if st.button('Show Curves'):
        fig = go.Figure()
        for model_id in selected_models:
            response = requests.get(f'http://fastapi:8000/get-curves/{model_id}')
            data = response.json().get('curves')
            fig.add_trace(go.Scatter(y=data['loss'], name=f'Model {model_id}'))
        st.plotly_chart(fig)
    
    # 4. Инференс
    st.subheader('Inference')
    uploaded_model = st.file_uploader('Upload a pre-trained model')
    if uploaded_model:
        # Logic for inference
        pass

