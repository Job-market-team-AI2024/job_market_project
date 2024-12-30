from typing import Dict, List
import pandas as pd
import numpy as np

field_dict = {
    'product': ['product', 'продукт', 'продакт'],
    'project': ['project', 'проект'],
    'data': ['data', 'дата', 'данных'],
    'bi': ['bi', 'би', 'визуализация'],
    'business': ['business', 'бизнес'],
    'system': ['system', 'системный'],
    'technical': ['qa', 'по', 'программного обеспечения', '1C', '1С', 'технический', 'technical',
                  'информационной безопасности'],
    'support': ['поддержки', 'поддержка', 'support'],
    'design': ['graphic', 'web', 'графический', 'веб']
}

role_dict = {
    'developer': ['developer', 'разработчик', 'программист', 'архитектор', 'architect', 'devops', 'mlops',
                  'разработка', 'разработку', 'программирование'],
    'scientist': ['scientist', 'science', 'саенс'],
    'analyst': ['analyst', 'analysis', 'analytics', 'аналитик'],
    'consultant': ['consultant', 'консультант', 'технолог'],
    'manager': ['manager', 'lead', 'owner', 'менеджер', 'лид', 'руководитель', 'руководителя', 'оунэр', 'оунер',
                'coordinator', 'координатор', 'директор', 'director', 'владелец', 'начальник', 'chief'],
    'tester': ['тестировщик', 'qa', 'автоматизатор тестирования', 'tester'],
    'engineer': ['engineer', 'инженер'],
    'specialist': ['specialist', 'operator', 'support', 'специалист', 'оператор', 'писатель', 'мастер', 'эксперт',
                   'поддержки', 'поддержка'],
    'designer': ['design', 'designer', 'дизайн', 'дизайнер', 'artist', 'художник'],
    'admin': ['администратор']
}

grade_dict = {
    'intern': ['intern', 'стажер'],
    'junior': ['junior', 'младший'],
    'middle': ['middle', 'ведущий'],
    'senior': ['senior', 'старший'],
    'lead': ['lead', 'руководитель', 'начальник']
}


def process_name(name: str, category_dict: Dict[str, List[str]]) -> str:
    for key, values in category_dict.items():
        for value in values:
            if value in name.lower():
                return key
    return 'other'


def preprocess_data(vacancies: List[Vacancy]) -> pd.DataFrame:
    df = pd.DataFrame([vacancy.dict() for vacancy in vacancies])
    df['field'] = df['name'].apply(lambda x: process_name(x, field_dict))
    df['role'] = df['name'].apply(lambda x: process_name(x, role_dict))
    df['grade'] = df['name'].apply(lambda x: process_name(x, grade_dict))
    df['salary'] = df[['salary_from', 'salary_to']].mean(axis=1)
    df['log_salary'] = np.log(df['salary'])
    return df

