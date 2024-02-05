import streamlit as st
from PIL import Image
import numpy as np
import grazpedwri_yolov8

#activate virtual enviroment: .venv\scripts\activate

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="LTLab - Pediatrics Wrist Fracture Detection")
    st.header('Pediatrics Radiograph Wrist Fracture Detection')
    #st.title('Pediatrics Wrist Fracture Detection')
    st.subheader("using YOLOv8.1 Algorithm on GRAZPEDWRI-DX")
    st.write("Special thanks to [Ultralytics](https://www.ultralytics.com/yolo) [ Rui-Yang Ju & Weiming Cai ](https://www.nature.com/articles/s41598-023-47460-7#Abs1) :pray:")
    st.write("### Try uploading an X-ray image for fracture detection")
    
    
    st.sidebar.write("## Upload a radiograph :gear:")
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    col1, col2 = st.columns(2)
    
    img_file = st.sidebar.file_uploader("Upload a radiograph",label_visibility="hidden", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        col1.write("Uploaded X-ray image :page_facing_up:")
        col1.image(img, img_file.name.split(".")[0])
    else:
        img = Image.open('data/default.jpg')
        col1.write("Default X-ray image :page_facing_up:")
        col1.image(img, caption="Distal forearm")

    placeholder = col2.empty()
    placeholder.write("Inferencing :running:")
    #col2.write("Inferencing :running:")
    predict_results = grazpedwri_yolov8.wrist_predict(img)
    #Save results to images
    grazpedwri_yolov8.results_to_img(predict_results)
    placeholder.write("Annotated fracture predition :male-detective:")
    #Result counts
    results_count = grazpedwri_yolov8.get_results_count(predict_results)
    results_count_text = grazpedwri_yolov8.get_results_text(results_count)
    col2.image("data/result_0.jpg",results_count_text)