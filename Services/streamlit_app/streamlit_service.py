import streamlit as st
import requests
# import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import json
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import numpy as np

### После отладки надо заменить локальный адрес на адрес, по которому стримлит прила будет искать эндпоинты при запуске контейнеров в докере
# API_BASE_URL = 'http://fastapi_app:8000'

### Локальная отладка
API_BASE_URL = 'http://127.0.0.1:8000'



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

    menu = st.selectbox('Menu', ['EDA', 'Create New Model', 'Get Model Info', 'Inference'])

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

        st.write(f'''Размерность данных: {uploaded_data.shape}''')

        st.subheader('Пропуски')
        missing_values_df = uploaded_data.isnull().sum().to_frame().reset_index().set_axis(
            ['Сolumn', 'Missing values count'], axis=1)
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
            if areas_dict[area_id]['parent_id'] is None or areas_dict[areas_dict[area_id]['parent_id']][
                'parent_id'] is None:
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

        st.write('Распределение вакансий по странам')
        st.dataframe(
            df['country_name'].value_counts().head(5).to_frame().reset_index().set_axis(['Country', 'Values count'],
                                                                                        axis=1))

        st.write('Распределение вакансий по городам в РФ')
        st.dataframe(
            df[df.country_name == 'Россия'].region_name.value_counts().head(5).to_frame().reset_index().set_axis(
                ['Country', 'Values count'], axis=1))

        st.write('Распределение вакансий по городам за пределами РФ')
        st.dataframe(
            df[df.country_name != 'Россия'].region_name.value_counts().head(5).to_frame().reset_index().set_axis(
                ['Country', 'Values count'], axis=1))

        st.subheader('Данные о профессиональной роли')

        st.dataframe(df['professional_roles_name'].value_counts().head(10).to_frame().reset_index().set_axis(
            ['Role', 'Values count'], axis=1))
        st.write('''Стоит отметить, что разбиение на основании названий не совсем корректно: например, аналитики могут быть бизнесовые, продуктовые, данных и т.д.,
        поэтому стоит рассмотреть распределение именно по функциональным ролям.''')

        product = ['product', 'продуктовый', 'продакт', 'продукта']
        project = ['project', 'проектов', 'проектный', 'проекта']
        data = ['data', 'дата', 'данных']
        bi = ['bi', 'би', 'визуализация']
        system = ['system', 'системный']
        business = ['business', 'бизнес']
        design = ['graphic', 'web', 'графический', 'веб']
        technical = ['qa', 'по', 'программного обеспечения', '1C', '1С', 'технический', 'technical',
                     'информационной безопасности']
        support = ['поддержки', 'поддержка', 'support']
        field = [
            ('product', product)
            , ('project', project)
            , ('data', data)
            , ('bi', bi)
            , ('business', business)
            , ('system', system)
            , ('technical', technical)
            , ('support', support)
            , ('design', design)
        ]

        engineer = ['engineer', 'инженер']
        developer = ['developer', 'разработчик', 'программист', 'архитектор', 'architect', 'devops', 'mlops',
                     'разработка', 'разработку', 'программирование']
        scientist = ['scientist', 'science', 'саенс']
        analyst = ['analyst', 'analysis', 'analytics', 'аналитик']
        consultant = ['consultant', 'консультант', 'технолог']
        manager = ['manager', 'lead', 'owner', 'менеджер', 'лид', 'руководитель', 'руководителя', 'оунэр', 'оунер',
                   'coordinator', 'координатор', 'директор', 'director', 'владелец', 'начальник', 'chief']
        designer = ['design', 'designer', 'дизайн', 'дизайнер', 'artist', 'художник']
        tester = ['тестировщик', 'qa', 'автоматизатор тестирования', 'tester']
        specialist = ['specialist', 'operator', 'support', 'специалист', 'оператор', 'писатель', 'мастер', 'эксперт',
                      'поддержки', 'поддержка']
        admin = ['администратор']
        role = [
            ('developer', developer)
            , ('scientist', scientist)
            , ('analyst', analyst)
            , ('consultant', consultant)
            , ('manager', manager)
            , ('tester', tester)
            , ('engineer', engineer)
            , ('specialist', specialist)
            , ('designer', designer)
            , ('admin', admin)
        ]

        intern = ['intern', 'стажер']
        junior = ['junior', 'младший']
        middle = ['middle', 'ведущий']
        senior = ['senior', 'старший']
        lead = ['lead', 'руководитель', 'начальник']
        grade = [
            ('intern', intern)
            , ('junior', junior)
            , ('middle', middle)
            , ('senior', senior)
            , ('lead', lead)
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

        st.write('Распределение с учётом функциональных ролей:')
        st.dataframe(df[['role', 'professional_roles_name']].value_counts().head(10).to_frame().reset_index().set_axis(
            ['Functional Role', 'Role', 'Values count'], axis=1))

        st.write('Распределение только функциональных ролей:')
        st.dataframe(
            df[['role']].value_counts().head(10).to_frame().reset_index().set_axis(['Functional Role', 'Values count'],
                                                                                   axis=1))

        st.subheader('Данные о профессиональных навыках')

        df['key_skills'] = df['key_skills'][~df['key_skills'].isnull()].str[1:-1].apply(
            lambda x: x.replace('"', '').lower().split(','))

        skills_counter = Counter(
            [skill for skill_list in df['key_skills'][df['key_skills'].notna()] for skill in skill_list])
        top_10_skills = skills_counter.most_common(10)
        top_10_skills_df = pd.DataFrame([(i[0].capitalize(), i[1]) for i in top_10_skills],
                                        columns=['Навык', 'Количество вакансий']).set_index('Навык')

        st.write('Топ-10 наиболее востребованных навыков:')
        st.dataframe(top_10_skills_df)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skills_counter)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.subheader('Данные о зарплате')

        st.dataframe(
            df['salary_currency'].value_counts().head(5).to_frame().reset_index().set_axis(['Currency', 'Values count'],
                                                                                           axis=1))
        df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1)

        st.write('''В большинстве вакансий, примерно в 93%, зарплата указана в рублях, в 3% – в тенге, также в редких 
        случаях встречаются белорусские рубли, евро и другие валюты. Ввиду нестабильности валютного курса и различных
        региональных особенностей рынков труда далее будем рассматривать только вакансии с указаниеем зарплаты в рублях.''')

        st.write('Cтатистика зарплат по ролям')

        st.dataframe(df[df['salary_currency'] == 'RUR'].groupby(['role']) \
                     .agg(avg_salary=('salary', 'mean'),
                          median_salary=('salary', 'median'),
                          count=('salary', 'nunique')) \
                     .sort_values(by='avg_salary', ascending=False) \
                     .round())

        st.write('Cтатистика зарплат по направлениям')

        st.dataframe(df[df['salary_currency'] == 'RUR'].groupby(['field']) \
                     .agg(avg_salary=('salary', 'mean'),
                          median_salary=('salary', 'median'),
                          count=('salary', 'nunique')) \
                     .sort_values(by='avg_salary', ascending=False) \
                     .round())

        st.write('Графики распределения рублевых зарплат с группировкой по кол-ву опыта с выбросами и без')

        experience_order = ['Нет опыта', 'От 1 года до 3 лет', 'От 3 до 6 лет', 'Более 6 лет']
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 8))
        RUR_salaries = df[df['salary_currency'] == 'RUR']
        IQR = RUR_salaries['salary'].quantile(0.75) - RUR_salaries['salary'].quantile(0.25)
        RUR_salaries_clipped = RUR_salaries[(RUR_salaries.salary > RUR_salaries.salary.quantile(0.25) - 1.5 * IQR) &
                                            (RUR_salaries.salary < RUR_salaries.salary.quantile(0.75) + 1.5 * IQR)]

        sns.boxplot(data=RUR_salaries,
                    y='salary',
                    hue='experience',
                    hue_order=experience_order,
                    ax=ax1)

        sns.boxplot(data=RUR_salaries_clipped,
                    y='salary',
                    hue='experience',
                    hue_order=experience_order,
                    ax=ax2)

        ax1.set_title('Распределение рублевых зарплат с группировкой по кол-ву опыта')
        ax1.set_ylabel('Размер рублевой зарплаты')
        ax1.set_xlabel('Кол-во опыта')
        ax1.grid()

        ax2.set_title('Распределение рублевых зарплат (без выбросов) с группировкой по кол-ву опыта')
        ax2.set_ylabel('Размер рублевой зарплаты')
        ax2.set_xlabel('Кол-во опыта')
        ax2.grid()

        st.pyplot(plt)

        st.write('Общие графики распределения зарплаты и логарифма зарплаты')

        df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1)
        df = df[~df['salary'].isnull()]
        df = df[df['salary_currency'] == 'RUR']
        df = df[df['country_name'] == 'Россия']
        old = df.shape[0]
        df = df[(df['salary'] > 10000)]  # np.quantile(df['salary'],0.005))]
        new = df.shape[0]

        df['log_salary'] = np.log(df['salary'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].hist(df['salary'], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title('Salary Distribution')
        axes[0].set_xlabel('Salary')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(df['log_salary'], bins=30, color='lightgreen', edgecolor='black')
        axes[1].set_title('Log-Transformed Salary Distribution')
        axes[1].set_xlabel('Log(Salary)')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(plt)

    if menu == 'Create New Model':

        st.header('Train a Model with Hyperparameters')

        st.subheader('Select Hyperparameter Values')
        fit_intercept = st.checkbox('Add fit_intercept?', value=True)
        normalize = st.checkbox('Normalise data?', value=False)

        selected_model_id = st.text_input('Model ID', value='new_model')

        hyperparameters = {
            'fit_intercept': fit_intercept,
            'normalize': normalize}

        if st.button(f'''Create Model {selected_model_id}'''):
            uploaded_data_mod = uploaded_data.replace(np.nan, None)
            active_payload = {
                'model_id': selected_model_id
            }
            active_response = requests.post(f'{API_BASE_URL}/set', json=active_payload)
            
            if active_response.status_code == 200:
                st.success('Model activated successfully')

            fit_payload = {
                            'model_id': selected_model_id,
                            'data': uploaded_data_mod.to_dict(orient='records')
                        }
            fit_response = requests.post(f'{API_BASE_URL}/fit', json=fit_payload)
            
            if fit_response.status_code == 200:
                st.success(f'''Model {model_id} fitted''')
            else:
                st.error(f"Error: {fit_response.status_code} - {fit_response.json().get('detail', 'Unknown error')}")

    
    if menu == 'Get Model Info':
        st.header('Information about model and learning curves')

        try:
            models_response = requests.get(f'{API_BASE_URL}/models')
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_list = [model['model_id'] for model in models_data['models']]

                if model_list:
                    selected_model_id = st.selectbox('Select Model to get information', model_list)
                    if st.button(f'Get information about "{selected_model_id}"'):

                        active_payload = {
                            'model_id': selected_model_id
                        }
                        active_response = requests.post(f'{API_BASE_URL}/set', json=active_payload)

                        if active_response.status_code == 200:
                            st.success('Model activated successfully')
                            
                        info_payload = {
                            'model_id': selected_model_id,
                        }
                        info_response = requests.post(f'{API_BASE_URL}/get', json=info_payload)

                        if info_response.status_code == 200:
                            model_info = info_response.json()

                            st.subheader(f"Model Info {selected_model_id}")
                            st.write("Коэффициенты:", model_info["coefficients"])
                            st.write("Интерсепт:", model_info["intercept"])

                            # Кривые обучения
                            st.subheader("Кривые обучения")
                            learning_curve_data = model_info["learning_curve"]
                            train_sizes = np.array(learning_curve_data["train_sizes"])
                            train_mean = np.array(learning_curve_data["train_mean"])
                            train_std = np.array(learning_curve_data["train_std"])
                            test_mean = np.array(learning_curve_data["test_mean"])
                            test_std = np.array(learning_curve_data["test_std"])

                            plt.figure(figsize=(10, 6))
                            plt.plot(train_sizes, train_mean, label="Train Score", marker='o')
                            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
                            plt.plot(train_sizes, test_mean, label="Test Score", marker='o')
                            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
                            plt.xlabel("Количество примеров обучения")
                            plt.ylabel("Оценка")
                            plt.title("Кривые обучения")
                            plt.legend(loc="best")
                            plt.grid()
                            st.pyplot(plt)
                            # st.dataframe(pd.DataFrame({'Prediction': predictions}))
                        else:
                            st.error(
                                f"Error: {info_response.status_code} - {info_response.json().get('detail', 'Unknown error')}")
                else:
                    st.warning('No models available.')
            else:
                st.error(
                    f"Error: {models_response.status_code} - {models_response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

    if menu == 'Inference':
        st.header('Make Predictions')

        try:
            models_response = requests.get(f'{API_BASE_URL}/models')
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_list = [model['model_id'] for model in models_data['models']]

                if model_list:
                    selected_model_id = st.selectbox('Select Model for Prediction', model_list)
                    if st.button(f'Make Predictions with "{selected_model_id}"'):
                        uploaded_data_mod = uploaded_data.replace(np.nan, None)
                        active_payload = {
                            'model_id': selected_model_id
                        }
                        active_response = requests.post(f'{API_BASE_URL}/set', json=active_payload)

                        if active_response.status_code == 200:
                            st.success('Model activated successfully')

                        prediction_payload = {
                            'model_id': selected_model_id,
                            'data': uploaded_data_mod.to_dict(orient='records')
                        }
                        prediction_response = requests.post(f'{API_BASE_URL}/predict', json=prediction_payload)

                        if prediction_response.status_code == 200:
                            predictions = prediction_response.json()['predictions']
                            uploaded_data['Prediction'] = predictions
                            st.success('Predictions generated successfully!')
                            st.dataframe(uploaded_data)
                        else:
                            st.error(
                                f"Error: {prediction_response.status_code} - {prediction_response.json().get('detail', 'Unknown error')}")
                else:
                    st.warning('No models available.')
            else:
                st.error(
                    f"Error: {models_response.status_code} - {models_response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")


