import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from pymilvus import connections, Collection
from tqdm import tqdm
from PIL import Image
from nn_functions import *
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel
import cv2
from googletrans import Translator

# инициализируем класс
translator = Translator()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)

app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
IMG_FOLDER = os.path.join("static", "photo")


# Загрузка данных из файла
def load_data_from_pkl():
    data = pd.read_pickle("static/Vid2Embd_50k_new.pkl")
    return data


def insert_video_to_pickle(link, tags, embd,ocr_text):
    # Путь к файлу данных
    try:
        # Загрузка существующих данных
        data = pd.read_pickle("static/Vid2Embd_50k_new.pkl")
        print("Мы в трае)))")
    except FileNotFoundError:
        # Если файл не существует, создаем новый DataFrame
        print("Мы в эксепте, оу нет!")
        data = pd.DataFrame(columns=['link', 'description', 'embds'])

    # Создание новой записи
    SpeechRec = ocr_text
    # tags_string = ', '.join(tags) if tags and isinstance(tags, list) else ''
    new_entry = pd.DataFrame([[link, tags, embd, SpeechRec]],
                             columns=['link', 'description', 'embds', 'SpeechRec'])
    print("new_entry['embd'].iloc[0].shape", new_entry['embds'].iloc[0].shape)
    # Добавление записи к существующим данным
    updated_data = pd.concat([data, new_entry], ignore_index=True)

    # Сохранение обновленных данных обратно в файл
    updated_data.to_pickle("static/Vid2Embd_50k_new.pkl")
    print("Data inserted successfully into pickle file.")


@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = request.form.get('search', '')
    ten_videos = []
    # ten_tags = []
    if search_query:
        search_query_eng = translator.translate(search_query).text
        videos = load_data_from_pkl()
        # df_50k = pd.read_pickle('Vid2Embd_50k.pkl').dropna(subset='embds')
        videos['score'] = model_clip.encode_text(search_query_eng) @ np.stack(videos['embds'].values).T
        # ten_videos = videos.sort_values(by='score', ascending=False)[['link', 'description']].head(15).values
        ten_videos = [list(i) for i in list(
            videos.sort_values(by='score', ascending=False)[['link', 'description']].head(15).fillna('').values)]
        eng_teg_video = [list(i) for i in list(
            videos[videos['description'].fillna('').str.contains(search_query_eng, regex=False)][['link', 'description']].head(10).values)]
        teg_video = [list(i) for i in list(
            videos[videos['description'].fillna('').str.contains(search_query, regex=False)][['link', 'description']].head(10).values)]
        combined_videos = pd.concat([eng_teg_video, teg_video]).drop_duplicates().head(10)


    return render_template('index.html', videos=combined_videos, search_query=search_query)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video_url = request.form.get('video-url')
        video_links_file = request.files.get('video-links-file')
        # video_title = request.form.get('video-title')
        video_tags = request.form.get('video-tags')

        if video_links_file:
            # Преобразование загруженного файла в JSON
            links_data = json.load(video_links_file)
            # Обработка каждой записи в JSON файле
            for item in links_data:
                link = item.get('link')
                tags = item.get('tags', None)  # Установка None если теги отсутствуют
                # Извлечение характеристик видео
                embedding = extract_video_features(link, model_clip)
                print("embedding.shape", embedding.shape)
                ocr_text = extract_text_from_video(link)
                # print("ocr_text", ocr_text)
                insert_video_to_pickle(link, tags, embedding, ocr_text)

        if video_url:
            embedding = extract_video_features(video_url, model_clip)
            insert_video_to_pickle(video_url, video_tags, embedding)

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
