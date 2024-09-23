import ultralytics
import torch
import cv2
import pandas as pd
from yolo_helper import convert_tracking_results_to_pandas
#from streamlit_webrtc import VideoTransformerBase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import av

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        """ Constructor """
        super().__init__()
        self.frame_counter = 0  # Keep track of frames processed
        self.results_df = pd.DataFrame()  # To store tracking results
        pretrained_model = "yolov8n.pt"
        self.model = ultralytics.YOLO(pretrained_model)
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def recv(self, frame):
        """
        Process each video frame using YOLOv8 model for object detection/tracking.
        """
        self.frame_counter += 1
        img = frame.to_ndarray(format="bgr24")

        # YOLO tracking
        results = self.model.track(source=img, conf=0.2, iou=0.6, stream=True, device=self.device)
        print(results)
        #
        
        # Convert tracking results to a DataFrame for further processing/plotting
        df = convert_tracking_results_to_pandas(results)
        self.results_df = pd.concat([self.results_df, df])

        img = df_to_image(self.results_df)
        frame_new=image_to_video_frame(img)
        return frame_new  # Return the original frame (or process if necessary)
    
    # def get_tracking_results(self):
    #     """ Get the accumulated YOLO tracking results as a DataFrame. """
    #     return self.results_df
    

# Step 1: Convert DataFrame to a heatmap image using Matplotlib
def df_to_image(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(df, cmap='viridis')
    fig.colorbar(cax)
    
    # Save it to a buffer
    fig.canvas.draw()
    
    # Convert the figure to a NumPy array (image)
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close the figure to avoid memory issues
    plt.close(fig)
    
    return img

# Step 2: Convert the image into a video frame using PyAV or OpenCV
def image_to_video_frame(img):
    # Convert the NumPy array (image) into an OpenCV-compatible format
    # In this case, the format is BGR for OpenCV, but it can be modified
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    height, width, _ = img_bgr.shape

    # Calculate the center point
    center_x = width // 2
    center_y = height // 2

    # Draw a red dot (circle) at the center of the image
    cv2.circle(img_bgr, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
    
    # Convert the BGR image to a VideoFrame using PyAV
    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# def streaming_call_back():
#     df_result = VideoProcessor.get_tracking_results()
#     # Convert the DataFrame to an image
#     img = df_to_image(df_result)
#     # Convert the image to a video frame
#     result_video_frame = image_to_video_frame(img)
#     return result_video_frame
