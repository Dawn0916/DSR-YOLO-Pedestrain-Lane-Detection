import threading

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame


def video_streaming_manipulation():
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

    ## Preparation for the frame manipulation  
    ############ will be replaced by pedestrain and lane detection ###############
    fig_place = st.empty()
    fig, ax = plt.subplots(1, 1)

    ## loop for the manipulation of each frame during the video streaming
    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue
        ###  Manipulate the current frame
        # ########## will be replaced by pedestrain and lane detection ###############
        # img = pedestrain_detection_manipulation(img)
        # img = lane_detection_manipulation(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ax.cla()
        ax.hist(gray.ravel(), 256, [0, 256])
        fig_place.pyplot(fig)


lock = threading.Lock()
img_container = {"img": None}
video_streaming_manipulation() 
