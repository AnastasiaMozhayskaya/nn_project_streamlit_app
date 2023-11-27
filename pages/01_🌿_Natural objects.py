import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import io
from model.preprocessing1 import preprocess
import numpy as np
import requests
import time
from io import BytesIO

idx6class = {0: "Здания",
             1: "Лес",
             2: "Ледник",
             3: "Горы",
             4: "Море",
             5: "Улица"}

st.set_page_config(
    page_title="Распознавание объектов природы"
)

st.markdown(
    """
    #### Это приложение на базе модели ResNet50 позволяет определить объекты природы, улицы и здания
"""
)

st.markdown("<p style='color: gray; font-size:20px'>Качество модели <em>Accuracy</em> составляет 81.2 %</p>", unsafe_allow_html=True)

st.markdown(
    """
    ###### Модель обучена распознавать 6 разновидностей объектов: Здания, Лес, Ледник, Горы, Море, Улица. Пожалуйста, \
    подайте на вход модели фотографию с одним из указанных объектов
"""
)

@st.cache_resource()
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 6)
    model.load_state_dict(torch.load('model/6classes_ResNet50.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(img):
    img = preprocess(img)
    pred = idx6class[model(img.unsqueeze(0)).argmax().item()]
    return pred

selected = st.radio('Способ загрузки', ['файл', 'ссылка URL'])
start_time = time.time()

if selected == 'файл':
    uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            resized_image = image.resize((448, 448))
            prediction = predict(resized_image)
            st.image(resized_image)
            st.write(f"<h2 style='font-size: 16px;'>Предсказанный класс: \
                            <span style='color:royalblue; font-size:17px'>{prediction}</span></h2>", unsafe_allow_html=True)
            st.subheader(f'Время предсказания: {round((time.time() - start_time), 2)} сек.')
else:
    image_urls = st.text_area('Введите URL-ы фотографий (разделяйте их новой строкой)')
    if image_urls:
        urls = image_urls.split('\n')
        for url in urls:
            if url.lower().endswith(('.jpg', '.jpeg', '.png')):
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                resized_image = image.resize((448, 448))
                prediction = predict(resized_image)
                st.image(resized_image)
                st.write(f"<h2 style='font-size: 16px;'>Предсказанный класс: \
                            <span style='color:royalblue; font-size:17px'>{prediction}</span></h2>", unsafe_allow_html=True)
                st.subheader(f'Время предсказания: {round((time.time() - start_time), 2)} сек.')
            
            else:
                st.write(f"Неправильный формат изображения: {url}. Пожалуйста, введите URL изображения с расширением .jpg, .jpeg или .png.")
