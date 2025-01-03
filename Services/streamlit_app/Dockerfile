### офф питон образ
FROM python:3.9-slim

### рабочая директория внутри контейнера
WORKDIR /app

### копируем все файлы в контейнер
COPY . .

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

### установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

### определяем порт связи с контейнером
EXPOSE 8501

### запуск приложения внутри контейнера
CMD ["streamlit", "run", "streamlit_service.py", "--server.port=8501", "--server.address=0.0.0.0"]