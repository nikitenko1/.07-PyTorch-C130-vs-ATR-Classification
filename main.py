import streamlit as st
from utils import set_background
from PIL import Image
import torch
from classifier import classify
from io import BytesIO
import base64
from model import CNN

st.set_page_config(
    page_title='C130 and ATR Classification',
    layout='centered'
)

set_background('utils/bg.jpeg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #000000;
        font-weight: bold;        
        margin-top: -65px;
    }
    .header {
        text-align: center;
        font-size: 35px;
        color: #2C2626;
        margin-top: -15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">C130 and ATR Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as an C130 and ATR</div>', unsafe_allow_html=True)

file = st.file_uploader('',type = ['jpg','jpeg','png','jfif'])

model = CNN()
model.load_state_dict(torch.load('c130_vs_atr.pth', weights_only=True))
model.eval()

class_names = {0:'C130',1:'ATR'}

if file is not None:
    
    image = Image.open(file).convert('RGB')
    
    prediction, score = classify(image, model, class_names)
    
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()

    # Display classification results with reduced gap and no extra space
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">        
        <img src="data:image/png;base64,{img_base64}" style="width: 350px; height: 320px; object-fit: cover;">
        <div style="font-size:40px; font-weight:bold; margin-left: 20px; color:#000000;">
            <p> <strong> Result: {prediction} </strong> </p>
            <p style="margin-top:-10px;"> <strong> Score: {score}% </strong> </p>
        </div>        
        </div>
        """,
        unsafe_allow_html=True
    )