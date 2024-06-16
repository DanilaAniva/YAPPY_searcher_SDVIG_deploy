# Использование официального образа Debian Bullseye как базового
FROM debian:bullseye-slim

# Установка Python и необходимых пакетов
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя без прав суперпользователя
RUN groupadd -r appuser && useradd -r -g appuser -G audio,video appuser \
    && mkdir -p /home/appuser/app \
    && chown -R appuser:appuser /home/appuser

# Установка рабочей директории в контейнере
WORKDIR /home/appuser/app

# Копирование файла с зависимостями в рабочую директорию
COPY requirements.txt .

# Установка зависимостей из файла requirements.txt под учетной записью пользователя
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование содержимого локальной директории в рабочую директорию контейнера
COPY --chown=appuser:appuser . .

# Пользователь, от которого будет запущен контейнер
USER appuser

# Открытие порта 5000 для внешнего мира
EXPOSE 5000

# Окружающая переменная, устанавливаемая средой выполнения
ENV PORT=5000

# Запуск приложения при старте контейнера
CMD ["python3", "app.py"]
