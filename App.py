import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO('best1.pt')



st.title("♻️ Waste Segregation Inference App")
st.markdown("Detect **Plastic**, **Paper**, **Metal**, and **Glass** using YOLO model.")

def infer_image(img):
    results = model(img)
    return results[0].plot()  # Draws bounding boxes with class labels

# Upload section
st.subheader("📂 Upload an Image or Video")
upload_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov"])

if upload_file is not None:
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)

    if upload_file.type.startswith("image"):
        image = cv2.imdecode(file_bytes, 1)
        result_img = infer_image(image)
        st.image(result_img, channels="BGR", caption="Detected Image with Labels")
    elif upload_file.type.startswith("video"):
        st.video(upload_file)
        tfile = open("temp_video.mp4", "wb")
        tfile.write(file_bytes)
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = infer_image(frame)
            stframe.image(result_frame, channels="BGR")
        cap.release()

# Live camera section
st.subheader("📸 Or use Live Camera")
if st.button("Start Live Camera"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    stop = st.button("Stop Camera")  # Dummy stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break
        result_frame = infer_image(frame)
        stframe.image(result_frame, channels="BGR")
        if stop:
            break

    cap.release()
