# src/app.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from detect_and_classify import detect_and_annotate

from PIL import Image

st.set_page_config(page_title="Visual Pollution Detector", layout="centered")
st.title("Visual Pollution Detector (MCVXAI-VPD)")
st.write("Upload an image — model will return same image with green boxes and labels.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    # Save uploaded file
    with open("tmp_uploaded.jpg","wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("tmp_uploaded.jpg", caption="Uploaded Image", use_column_width=True)
    if st.button("Run detection"):
        with st.spinner('Detecting...'):
            out = detect_and_annotate("tmp_uploaded.jpg", "tmp_out.jpg")
        st.success("Done")
        st.image("tmp_out.jpg", caption="Detected (green boxes)", use_column_width=True)
