### офф питон образ
FROM python:3.9-slim

### рабочая директория внутри контейнера
WORKDIR /app

### копируем все файлы в контейнер
COPY . .

### установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

### определяем порт связи с контейнером
EXPOSE 8000

### запуск приложения внутри контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

