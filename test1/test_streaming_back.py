import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("My first Streamlit app")
st.write("Hello, world")

threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)

    # Get image dimensions
    height, width, _ = img.shape

    # Calculate the center point
    center_x = width // 2
    center_y = height // 2

    # Draw a red dot (circle) at the center of the image
    cv2.circle(img, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback) ## callback is a good start!