from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel
import cv2
import easyocr


def extract_video_features(url, model, frame_interval=10):
    '''
    Извлекает эмбеддинги из видео, выбирая кадры с заданным интервалом.

    Данная функция загружает видео по указанной URL, преобразует избранные кадры в формат PIL Image,
    и использует предварительно обученную модель для получения эмбеддингов изображений. Финальный
    эмбеддинг видео получается путём агрегации эмбеддингов отдельных кадров с использованием операции
    минимума по оси.

    Параметры:
        url (str): Ссылка на видео.
        frame_interval (int, optional): Интервал между обрабатываемыми кадрами.
                                        По умолчанию равен 10.

    Возвращаемое значение:
        np.array: Массив с минимальным эмбеддингом среди обработанных кадров,
                  или None, если видео не содержит кадров.

    Пример использования:
        video_url = "http://example.com/video.mp4"
        embedding = extract_video_features(video_url, frame_interval=5)
    '''

    # Инициализация захвата видео с использованием OpenCV
    cap = cv2.VideoCapture(url)
    frames = []  # список для хранения обработанных кадров
    frame_count = 0

    # Чтение первого кадра
    success, frame = cap.read()

    # Цикл по всем кадрам видео
    while success:
        # Обработка кадра, если он соответствует заданному интервалу
        if frame_count % frame_interval == 0:
            # Преобразование кадра из BGR в RGB и затем в объект PIL Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)

        frame_count += 1
        success, frame = cap.read()  # Чтение следующего кадра

    # Освобождение ресурсов захвата видео
    cap.release()

    # Если были обработаны кадры, получаем эмбеддинги и возвращаем их агрегацию
    if frames:
        with torch.no_grad():
            # Получение эмбеддингов для каждого кадра с помощью модели
            image_embeddings = model.encode_image(frames)
        # Возвращение минимального эмбеддинга
        return np.min(image_embeddings, axis=0)
    else:
        # Возвращение None, если кадры отсутствуют
        return None


def extract_text_from_video(video_path, interval=120, confidence_threshold=0.6):
    """
    Извлекает текст из видео с использованием OCR.

    Параметры:
    video_path (str): URL или путь к видеофайлу.
    interval (int): Интервал между обрабатываемыми кадрами (по умолчанию каждые 30 кадров).
    confidence_threshold (float): Минимальная вероятность для включения текста (по умолчанию 0.6).

    Возвращает:
    str: Объединённый текст из всех кадров.
    """

    # Инициализация видео захвата
    capture = cv2.VideoCapture(video_path)

    # Инициализация OCR ридера из библиотеки easyocr
    reader = easyocr.Reader(['ru', 'en'])  # Указываем языки, для которых необходимо распознавание

    frame_count = 0  # Счетчик кадров
    final_texts = []  # Список для хранения всех распознанных строк текста
    seen_texts = set()  # Множество для хранения строк, которые уже были добавлены

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Обработка каждого предзаданного кадра
        if frame_count % interval == 0:
            # Применение OCR для текущего кадра
            result = reader.readtext(frame)

            # Вывод распознанного текста и вероятностей для текущего кадра
            for detection in result:
                (_, text, prob) = detection

                # Фильтрация текста по вероятности
                if prob >= confidence_threshold:
                    # Проверка на часть обнаруженного текста в ранее извлеченных
                    if not any(text in seen for seen in seen_texts):
                        final_texts.append(text)
                        seen_texts.add(text)

        frame_count += 1

    # Освобождение video capture
    capture.release()

    # Объединение всех распознанных строк в один текст
    final_text = ' '.join(final_texts)
    return final_text
