import os
import io
from typing import Callable
import cv2
import uuid
import pandas as pd
#import tempfile
import ultralytics
import torch
from yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas

#import av
#import subprocess
import moviepy.editor as moviepy

class VideoHandler():

    def __init__(self, video: bytes):
        """ Constructor """
        self.byte_video = io.BytesIO(video.read())
        self.temp_id = str(uuid.uuid4())
        self.video_path = self.temp_id + ".mp4"

        #write file to given video_path
        with open(self.video_path, 'wb') as out:
            out.write(self.byte_video.read())
        
    def get_video_path(self):
        return self.video_path
    
    # def get_tracked_video_file(self)

    def get_video_stats(self) -> tuple:
        """
        Retrieve statistics about a video file.

        Returns:
        - tuple: A tuple containing video statistics in the following format:
            (fps (int), width (int), height (int), total_frames (int))
        """
        vf = cv2.VideoCapture(self.video_path)

        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))

        return (fps, width, height, total_frames)
    
    def track(self,progressbar_callback: Callable) -> tuple[pd.DataFrame, str]:
        """
        Perform object tracking on the video with YOLOv8.
        Args:
            progressbar_callback (Callable[int]): a callback accepting 1 argument (frame number)
        Return:
            DataFrame with tracking results
            Processed video path
        """
        ## Loading the Pretrained YOLOv8 Model
        pretrained_model = "yolov8n.pt"   # Loads the lightweight YOLOv8 model (yolov8n.pt). 
        model = ultralytics.YOLO(pretrained_model, verbose=True) # Initializes the YOLO model from the ultralytics library. The verbose=True argument makes sure the model prints progress and status messages while loading and running.

        ## Setting Up Progress Reporting
        yolo_progress_reporting_event = "on_predict_batch_start"
        progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
                                                                       lambda _,counter: progressbar_callback(counter))
        model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)
        
        ## Configuring Device for YOLOv8
        device = 0 if torch.cuda.is_available() else 'cpu' 

        ## Setting Output Directory
        outputdir=os.getcwd()
        print(f"Using outputdir: {outputdir}")

        ## Tracking Objects in the Video
        tracking_results = model.track(source=self.video_path, conf=0.2, iou=0.6, show=False, device=device, stream=True, save=True, save_dir=outputdir, exist_ok=True, project=outputdir)
        print(f"Tracking results from ultralytics.YOLO.track function:{tracking_results}")  # What is the data type???? 


        ## Converting Tracking Results to a DataFrame
        results_df = convert_tracking_results_to_pandas(tracking_results)



        ## load the processed video and export it to a new file
        processed_video_path = os.path.join(outputdir, "track", self.temp_id + ".avi")
        converted_video_path = os.path.join(outputdir, "track", self.temp_id + ".mp4")
        print(f"Converting the output ({processed_video_path}) to the format readable by streamlit")
  
        clip = moviepy.VideoFileClip(processed_video_path) # Loading a Video
        clip.write_videofile(converted_video_path)  #  Writing/Exporting the Video
        """
        How It Works:
        Loading a Video: clip = moviepy.VideoFileClip(processed_video_path)

        This line loads a video file from processed_video_path (e.g., "processed_video.mp4") into a VideoFileClip object called clip.
        Writing/Exporting the Video: clip.write_videofile(converted_video_path)

        This line saves the clip object (which contains the video data) to a new file path specified by converted_video_path (e.g., "converted_video.mp4").
        """

        print(f"Conversion complete")
        return results_df, converted_video_path
    
    def __del__(self):
        """
        Remove a video file.
        """
        os.remove(self.video_path)
