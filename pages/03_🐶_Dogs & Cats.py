import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import io
from model.preprocessing import preprocess
import numpy as np
import requests
from io import BytesIO

device="cpu" 
idx2class = {0: 'Кошка', 1: 'Собака'}

st.set_page_config(
    page_title="Распознавание кошек и собак"
)

st.markdown(
    """
    #### Это приложение на базе модели ResNet18 позволяет определить один из двух классов домашнего питомца Кошка/Собака по фотографиям 
"""
)

st.markdown("<p style='color: gray; font-size:20px'>Качество модели <em>Accuracy</em> составляет 97.5 %</p>", unsafe_allow_html=True)

@st.cache_resource()
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load('model/best_model_ResNet18.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(img):
    img = preprocess(img)
    pred = idx2class[torch.sigmoid(model(img.unsqueeze(0))).round().item()]
    return pred


selected = st.radio('Способ загрузки', ['файл', 'ссылка URL'])

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
            
            else:
                st.write(f"Неправильный формат изображения: {url}. Пожалуйста, введите URL изображения с расширением .jpg, .jpeg или .png.")