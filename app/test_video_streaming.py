import os
import cv2
import uuid
import pandas as pd
import ultralytics
import torch
from yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas
import moviepy.editor as moviepy
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading

lock = threading.Lock()
img_container = {"img": None}


class VideoHandler(VideoTransformerBase):
    def __init__(self):
        """ Constructor """
        self.temp_id = str(uuid.uuid4())
        self.frame_counter = 0
        self.results_df = pd.DataFrame()  # To store tracking results
        
        # YOLO model initialization
        pretrained_model = "yolov8n.pt"
        self.model = ultralytics.YOLO(pretrained_model, verbose=True)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"YOLO initialized on {'GPU' if self.device == 0 else 'CPU'}")

    def transform(self, frame):
        """Process each video frame in real-time from the stream."""
        img = frame.to_ndarray(format="bgr24")
        
        with lock:
            img_container["img"] = img

        # Perform YOLO tracking every N frames (to avoid excessive processing)
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:  # Adjust frame skip as necessary
            tracking_results = self.model.track(source=img, conf=0.2, iou=0.6, stream=True, device=self.device)
            self.results_df = convert_tracking_results_to_pandas(tracking_results)
            # Optionally visualize or log tracking results
            print(f"Tracking results at frame {self.frame_counter}: {self.results_df}")

        return frame  # Return the unmodified frame to continue streaming


def video_streaming_manipulation():
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoHandler)

    # Preparation for frame manipulation (displaying stats, etc.)
    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue

        # Optional: process the frame further (e.g., convert to grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Processed Grayscale Frame", use_column_width=True)

        # Optionally, display YOLO statistics, if available
        st.write(f"Current tracking results: {ctx.video_transformer.results_df}")


# Streamlit interface for video streaming
st.title("YOLOv8 Video Streaming with Object Tracking")
video_streaming_manipulation()
