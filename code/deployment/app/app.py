import streamlit as st
import os
import requests

url = os.environ.get('API_URI')
print(url)

st.title('Модель предсказания')

features = st.text_input('Введите признаки через запятую')

if st.button('Предсказать'):
    try:
        features_list = list(map(float, features.split(',')))
        response = requests.post(f'http://{url}:8000/predict', json={'features': features_list})
        prediction = response.json().get('prediction')
        st.write(f'Предсказание: {prediction}')
    except Exception as e:
        st.write(f'Ошибка: {e}')