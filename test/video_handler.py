import ultralytics
import torch
import cv2
import pandas as pd
from yolo_helper import convert_tracking_results_to_pandas
#from streamlit_webrtc import VideoTransformerBase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        """ Constructor """
        super().__init__()
        self.frame_counter = 0  # Keep track of frames processed
        self.results_df = pd.DataFrame()  # To store tracking results
        pretrained_model = "yolov8n.pt"
        self.model = ultralytics.YOLO(pretrained_model)
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def transform(self, frame):
        """
        Process each video frame using YOLOv8 model for object detection/tracking.
        """
        self.frame_counter += 1
        img = frame.to_ndarray(format="bgr24")

        # YOLO tracking
        results = self.model.track(source=img, conf=0.2, iou=0.6, stream=True, device=self.device)

        # Convert tracking results to a DataFrame for further processing/plotting
        df = convert_tracking_results_to_pandas(results)
        self.results_df = pd.concat([self.results_df, df])

        return frame  # Return the original frame (or process if necessary)
    
    def get_tracking_results(self):
        """ Get the accumulated YOLO tracking results as a DataFrame. """
        return self.results_df
