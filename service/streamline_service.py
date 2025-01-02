import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import json
import logging
from logging.handlers import RotatingFileHandler
from collections import Counter
from wordcloud import WordCloud


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

uploaded_file = st.file_uploader('Please first upload your dataset')

if uploaded_file: 
    uploaded_data = pd.read_csv(uploaded_file)
    st.dataframe(uploaded_data.head())
    logger.info('Dataset uploaded successfully.')
    
    menu = st.sidebar.selectbox('Menu', ['EDA', 'Train Model & Learning Curves', 'Inference'])

    if menu == 'EDA':
        st.header('Exploratory Data Analysis')

        st.write('''Датасет включает в себя данные об IT-вакансиях, размещенных на портале Headhunter 
        c 18-го сентября 2023 года по 17-ое октября 2023. По каждой из вакансий присутствуют следующие 
        даннные:
        
        Описание вакансии
        name - название вакансии
        description - текстовое описание вакансии на HeadHunter
        schedule - тип рабочего графика
        professional_roles_name - профессиональная категория согласно HeadHunter
        published_at - дата публикации вакансии
        
        Зарплата
        salary_from - нижняя граница вилки зарплаты
        salary_to - верхняя граница вилки зарплаты
        salary_gross - индикатор, если зарплата указана в размере gross
        salary_currency - валюта зарплаты
        
        Требования к кандидату
        experience - требуемый опыт для вакансии
        key_skills - требуемые навыки
        languages - требуемое владение иностранными языками
        
        Работодатель
        employer_name - название работодателя
        accredited_it_employer - индикатор для аккредитованных в России IT-компаний
        
        Место работы
        area_name - названия населенного пункта, в котором размещена вакансия
        addres_raw - адрес места работы
        addres_lat - широта места работы
        address_lng - долгота места работы

        Целевая переменная - предлагаемая зарплата''')

        st.write(f'''Размерность данных: {uploaded_data.shape()}''')

        st.subheader('Пропуски')
        missing_values_df = uploaded_data.isnull().sum().to_frame().reset_index().set_axis(['Сolumn', 'Missing values count'], axis = 1)
        missing_values_df = missing_values_df[missing_values_df['Missing values count'] > 0]
        st.dataframe(missing_values_df)

        df = uploaded_data.copy()

        st.subheader('Дубликаты')

        st.write(f'''Количество полных дубликатов: {df.duplicated().sum()}''')

        st.subheader('Данные о месте работы')

        def area_transform(entry):
            areas_dict = {}
            areas_dict[entry['id']] = {'name': entry['name'], 'parent_id': entry['parent_id']}
            for area in entry['areas']:
                areas_dict.update(area_transform(area))
            return areas_dict
        
        def area_region(area_id, areas_dict):
            if areas_dict[area_id]['parent_id'] is None or areas_dict[areas_dict[area_id]['parent_id']]['parent_id'] is None:
                return areas_dict[area_id]['name']
            else:
                return areas_dict[areas_dict[area_id]['parent_id']]['name']
        
        def area_country(area_id, areas_dict):
            while areas_dict[area_id]['parent_id'] is not None:
                area_id = areas_dict[area_id]['parent_id']
            return areas_dict[area_id]['name']

        areas = requests.get('https://api.hh.ru/areas').json()

        areas_dict = {}
        
        for area in areas:
            areas_dict.update(area_transform(area))

        df['region_name'] = df['area_id'].apply(lambda x: area_region(str(x), areas_dict))
        df['country_name'] = df['area_id'].apply(lambda x: area_country(str(x), areas_dict))

        st.dataframe(df['country_name'].value_counts().head(5).to_frame().reset_index().set_axis(['Country', 'Values count'], axis = 1))
        st.write('Большинство вакансий (93 %) размещены в России. Относительно небольшое число вакансий также опубликованы в Казахастане, Беларуси, Узбекистане и других странах.')

        st.dataframe(df[df.country_name == 'Россия'].region_name.value_counts().head(5).to_frame().reset_index().set_axis(['Country', 'Values count'], axis = 1))
        st.write('Среди городов размещения в РФ вакансий лидирует Москва (41 %), Санкт-Петербург (11 %), Екатеринбург, Новосибирск и Казань.')

        st.dataframe(df[df.country_name != 'Россия'].region_name.value_counts().head(5).to_frame().reset_index().set_axis(['Country', 'Values count'], axis = 1))
        st.write('За пределами РФ лидируют столицы стран СНГ: Казахстана, Беларуси, Узбекистана')

        st.subheader('Данные о профессиональной роли')

        st.dataframe(df['professional_roles_name'].value_counts().head(10).to_frame().reset_index().set_axis(['Role', 'Values count'], axis = 1))
        st.write('''Наиболее популярные профессии в датасете – это программист/разработчик, специалист техничской поддержки и аналитик
        При этом стоит отметить, что данное разбиение на основании названий не совсем корректно: например, аналитики могут быть бизнесовые, продуктовые, данных и т.д.,
        поэтому стоит также рассмотреть распределение именно по функциональным ролям.''')
        
        product = ['product','продуктовый','продакт','продукта']
        project = ['project','проектов','проектный','проекта']
        data = ['data','дата','данных']
        bi = ['bi','би','визуализация']
        system = ['system','системный']
        business = ['business','бизнес']
        design = ['graphic','web','графический','веб']
        technical = ['qa','по','программного обеспечения','1C','1С','технический','technical','информационной безопасности']
        support = ['поддержки','поддержка','support']
        field = [
            ("product", product)
            ,("project", project)
            ,("data", data)
            ,("bi", bi)
            ,("business", business)
            ,("system", system)
            ,("technical", technical)
            ,("support", support)
            ,("design", design)
            ]
        
        engineer = ['engineer','инженер']
        developer = ['developer','разработчик','программист','архитектор','architect','devops','mlops','разработка','разработку','программирование']
        scientist = ['scientist','science','саенс']
        analyst = ['analyst','analysis','analytics','аналитик']
        consultant = ['consultant','консультант','технолог']
        manager = ['manager','lead','owner','менеджер','лид','руководитель','руководителя','оунэр','оунер','coordinator','координатор','директор','director','владелец','начальник','chief']
        designer = ['design','designer','дизайн','дизайнер','artist','художник']
        tester = ['тестировщик','qa','автоматизатор тестирования','tester']
        specialist = ['specialist','operator','support','специалист','оператор','писатель','мастер','эксперт','поддержки','поддержка']
        admin = ['администратор']
        role = [
            ("developer", developer)
            ,("scientist", scientist)
            ,("analyst", analyst)
            ,("consultant", consultant)
            ,("manager", manager)
            ,("tester", tester)
            ,("engineer", engineer)
            ,("specialist", specialist)
            ,("designer", designer)
            ,("admin", admin)
            ]
        
        intern = ['intern', 'стажер']
        junior = ['junior', 'младший']
        middle = ['middle', 'ведущий']
        senior = ['senior', 'старший']
        lead = ['lead', 'руководитель', 'начальник']
        grade = [
            ("intern", intern)
            ,("junior", junior)
            ,("middle", middle)
            ,("senior", senior)
            ,("lead", lead)
            ]

    def find_categories(name, categories):
        result = []
        for category, elements in categories:
            if any(el.lower() in name.lower() for el in elements):
                result.append(category)
        return result

    df['fields'] = df['name'].apply(lambda x: find_categories(x, field))
    df['roles'] = df['name'].apply(lambda x: find_categories(x, role))
    df['grades'] = df['name'].apply(lambda x: find_categories(x, grade))
    df['field'] = df['fields'].apply(lambda x: x[0] if x else 'other')
    df['role'] = df['roles'].apply(lambda x: x[0] if x else 'other')
    df['grade'] = df['grades'].apply(lambda x: x[0] if x else 'other')

    st.write('Вот распределение с учётом функциональных ролей:')
    st.dataframe(df[['role','professional_roles_name']].value_counts().head(10).to_frame().reset_index().set_axis(['Functional Role', 'Role', 'Values count'], axis = 1))

    st.write('А вот распределение только функциональных ролей:')
    st.dataframe(df[['role']].value_counts().head(10).to_frame().reset_index().set_axis(['Functional Role', 'Values count'], axis = 1))

    st.subheader('Данные о профессиональных навыках')

    df['key_skills'] = df['key_skills'][~df['key_skills'].isnull()].str[1:-1].apply(lambda x: x.replace('"', '').lower().split(','))

    skills_counter = Counter([skill for skill_list in df['key_skills'][df['key_skills'].notna()] for skill in skill_list])
    top_10_skills = skills_counter.most_common(10)
    top_10_skills_df = pd.DataFrame([(i[0].capitalize(), i[1]) for i in top_10_skills], columns=['Навык', 'Количество вакансий']).set_index('Навык')
    
    st.write('Топ-10 наиболее востребованных навыков:')
    st.dataframe(top_10_skills_df)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skills_counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.write('''Как видно, часто ищут вакансии со знаниями SQL, Linux, Git и Python; помимо этого работодатели обращают внимание на софт-скиллы: 
    умение работать в команде, аналитически мыслить и грамотно выращать свои мысли.''')

    st.subheader('Данные о зарплате')

    st.dataframe(df['salary_currency'].value_counts().head(5).to_frame().reset_index().set_axis(['Currency', 'Values count'], axis = 1))
    st.write('''В большинстве вакансий, примерно в 93%, зарплата указана в рублях, в 3% – в тенге, также в редких 
    случаях встречаются белорусские рубли, евро и другие валюты. Ввиду нестабильности валютного курса и различных
    региональных особенностей рынков труда далее будем рассматривать только вакансии с указаниеем зарплаты в рублях.''')

    df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1)
    
    df = df[~df['salary'].isnull()]
    df = df[df['salary_currency'] == 'RUR']
    df = df[df['country_name'] == 'Россия']
    old = df.shape[0]
    df = df[(df['salary'] > 10000)] #np.quantile(df['salary'],0.005))]
    new = df.shape[0]

    df['log_salary'] = np.log(df['salary'])
    fig, axes = plt.subplots(1,2, figsize = (12,6))
    axes[0].hist(df['salary'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Salary Distribution')
    axes[0].set_xlabel('Salary')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df['log_salary'], bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_title('Log-Transformed Salary Distribution')
    axes[1].set_xlabel('Log(Salary)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
        
    if menu == 'Train Model & Learning Curves':
        st.header('Train a Model with Hyperparameters')
        
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
