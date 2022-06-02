import time

import cv2
from cv2 import *
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 3

def streamlit_menu(example=3):

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "About Project", "Contacts"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 3. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "About Project", "Contacts"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"background-color": "#f8edeb"},
                "icon": {"color": "#ffb5a7"},
                "nav-link": {
                    # "font-size": "25px",
                    # "text-align": "left",
                    # "margin": "0px",
                    "--hover-color": "#fcd5ce",
                },
                "nav-link-selected": {"background-color": "#f5cac3"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
    # st.title(f"You have selected {selected}")
    # img_home = Image.open("images/image_2022-05-20_13-56-31.png")
    # st.image(img_home, width=1000)
    def load_lottieurl2(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_coding2 = load_lottieurl2("https://assets9.lottiefiles.com/packages/lf20_j6nmheu0.json")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader('Welcome to our website!')
            st.title('Hey, predict Age & Gender')
            st.write('Do you want to know how old your girlfriend is if she hides her age? Or maybe you just forgot how old your child is? Then this site will help you find out how old you and your friends look. All you need to do is just upload a photo and the model will do everything for you.')
        st.write("---")
        with right_column:
            st.write("##")
            st_lottie(lottie_coding2, height=300, key="coding")

    def get_face_box(net, frame, conf_threshold=0.7):
        opencv_dnn_frame = frame.copy()
        frame_height = opencv_dnn_frame.shape[0]
        frame_width = opencv_dnn_frame.shape[1]
        blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
            104, 117, 123], True, False)

        net.setInput(blob_img)
        detections = net.forward()
        b_boxes_detect = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                b_boxes_detect.append([x1, y1, x2, y2])
                cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                              (0, 255, 0), int(round(frame_height / 150)), 8)
        return opencv_dnn_frame, b_boxes_detect


    # st.title("""Try out Prediction!""")

    # st.subheader("""Upload a picture that contains a face: """)

    uploaded_file = st.file_uploader("Choose a file: ")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        cap = np.array(image)
        cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2BGRA))
        cap = cv2.imread("temp.jpg")

        face_txt_path = "opencv_face_detector.pbtxt"
        face_model_path = "opencv_face_detector_uint8.pb"

        age_txt_path = "age_deploy.prototxt"
        age_model_path = "age_net.caffemodel"

        gender_txt_path = "gender_deploy.prototxt"
        gender_model_path = "gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        age_classes = ['~1-2', '~3-5', '~6-15', '~16-22', '~23-30', '~31-40', '~41-60', 'age is greater than 60']
        gender_classes = ['Male', 'Female']

        age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
        gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
        face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

        padding = 20

        t = time.time()

        faces = []

        frameFace, b_boxes = get_face_box(face_net, cap)

        if not b_boxes:
            st.write("No face Detected, Checking next frame")

        for bbox in b_boxes:
            face = cap[max(0, bbox[1] - padding): min(bbox[3] + padding, cap.shape[0] - 1),
                   max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred_list = gender_net.forward()
            gender = gender_classes[gender_pred_list[0].argmax()]
            # st.write(f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

            age_net.setInput(blob)
            age_pred_list = age_net.forward()
            age = age_classes[age_pred_list[0].argmax()]
            # st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            faces.append(frameFace)

        st.image(faces[0])
        print(faces)

if selected == "About Project":
    local_css("style/style3.css")

    # ---- LOAD ASSETS ----
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_mbig1rjb.json")

    # ---- HEADER SECTION ----
    with st.container():
        st.title("Hi, we are Cool Cuties :wave:")
        st.header("And that's our Capstone Project: Age & Gender Prediction")
        st.write("The goal is to develop a webpage, with a help of which we will be able to download a photo of the human on the page, and later as an output the gender and age range will be gained. ")
        st.write("The human face carries a ton of knowledge, including identity, expression, emotion, gender, age, and so on. With the rapid rise of intelligent applications, there is indeed a growing demand for automatic facial attribute extraction. Age is one of the most essential facial characteristics in social interactions. As a result, automatic age estimation from face photos is a challenging and crucial topic that is being researched in a variety of applications, including access control, human-computer interaction (HCI), police departments, and monitoring.")

    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            # st.write("##")
            st.header("Background / About Dataset")
            st.write("""
            We have trained the model with [this dataset](https://talhassner.github.io/home/projects/Adience/)
             - Total number of photos: 26,580
             - Total number of subjects: 2,284
             - Number of age groups / labels: 8
             - (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-100)
             These images represent some of the challenges of age and gender estimation from real-world, unconstrained images. Most notably, extreme blur (low-resolution), occlusions, out-of-plane pose variations, expressions and so on.
             """)
            # st.write("[YouTube Channel >](https://youtube.com/)")
        with right_column:
            st.write("##")
            st_lottie(lottie_coding, height=300, key="coding")

    st.write("---")
    st.header("About us")
    st.write('We are students at Astana IT University, currently studying in group BD-2008. We have created this website as a project work for the discipline named "Capstone Project". Hope you will enjoy using it!')

    title = '<p style="text-align: center; font-weight: bold; font-size: 32px">Members of the group "Cool Cuties"</p>'
    st.markdown(title, unsafe_allow_html=True)
    with st.container():
        first_column, second_column, third_column, fourth_column = st.columns(4)
        with first_column:
            img_home = Image.open("images/IMG_6077.png")
            st.image(img_home)
            st.write("Ayazhan Sydyk")
            st.caption("the main software-tester cutie")
            # ayazhan_title = '<p style="text-align: center; font-weight: bold; font-size: 18px;>Ayazhan Sydyk</p>'
            # st.markdown(ayazhan_title, unsafe_allow_html=True)
            # ayazhan_wr = '<p style="font-weight: bold; font-size: 14px;>the main software-tester cutie</p>'
            # st.markdown(ayazhan_wr, unsafe_allow_html=True)
        with second_column:
            img_home = Image.open("images/IMG_6078.png")
            st.image(img_home)
            st.write("Yerkegul Assaiyn")
            st.caption("the main front-end cutie")
        with third_column:
            img_home = Image.open("images/IMG_6079.png")
            st.image(img_home)
            st.write("Assylnur Lesken")
            st.caption("the main model-builder cutie")
        with fourth_column:
            img_home = Image.open("images/IMG_6080.png")
            st.image(img_home)
            st.write("Alua Onayeva")
            st.caption("the main technical-writer cutie")


if selected == "Contacts":
    st.header(":mailbox: Get In Touch With Us!")

    contact_form = """
    <form action="https://formsubmit.co/assaiynerkegul@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)
    local_css("style/style.css")


hide_streamlit_style = """
             <style>
             #MainMenu {visibility: hidden;}
             footer {visibility: hidden;}
             body {background-color: pink;}
             </style>
             """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)