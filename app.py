import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Phone Price Predictor')

# Battery
battery = st.number_input('Battery (mAh)')

# Processor
processor = st.selectbox('Processor', df['Processor'].unique())

# RAM
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 20])

# ROM
rom = st.selectbox('ROM (GB)', [8, 16, 32, 64, 128, 256, 512, 1080])

# Main Camera
main_camera = st.number_input('Main Camera (Pixel)')

# Front Camera
front_camera = st.number_input('Front Camera (Pixel)')

# Company
company = st.selectbox('Company', df['Company'].unique())

# Color
color = st.selectbox('Color', df['Color'].unique())

# Model
model = st.selectbox('Model', df['Model'].unique())

if st.button('Predict Price'):
    # query
    query = np.array([battery, processor, ram, rom, main_camera, front_camera, company, color, model])

    query = query.reshape(1, 9)
    st.title('The predicted price of this configuration is ' + str(int(pipe.predict(query))))