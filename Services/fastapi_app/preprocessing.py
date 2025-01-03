from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Any, Tuple


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

col_from_name = {'field':field_dict,
                 'role':role_dict,
                 'grade':grade_dict}

def preprocess_name(name: str, category_dict: Dict[str, List[str]]) -> str:
    for key, values in category_dict.items():
        for value in values:
            if value in name.lower():
                return key
    return 'other'

class CustomPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_get_from_name=None):
        self.cols_to_get_from_name = cols_to_get_from_name if cols_to_get_from_name else []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.cols_to_get_from_name:
            for col in self.cols_to_get_from_name:
                df[col] = df['name'].apply(lambda x: preprocess_name(x, col_from_name[col]))
            # df = df.drop(self.cols_drop, axis=1, errors="ignore")
        return df
