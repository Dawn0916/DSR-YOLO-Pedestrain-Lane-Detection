import os
import io
from typing import Callable
import cv2
import uuid
import pandas as pd
import ultralytics
import torch
from yolo_helper import convert_tracking_results_to_pandas
from lane_detection import LaneDetector

class VideoHandler:
    def __init__(self, video: bytes):
        """ Constructor """
        self.byte_video = io.BytesIO(video.read())
        self.temp_id = str(uuid.uuid4())
        self.video_path = self.temp_id + ".mp4"

        # Initialize lane detector
        self.lane_detector = LaneDetector()

        # Write the video file to disk
        with open(self.video_path, 'wb') as out:
            out.write(self.byte_video.read())

    def get_video_path(self):
        return self.video_path

    def get_video_stats(self) -> tuple:
        """Retrieve statistics about the video."""
        vf = cv2.VideoCapture(self.video_path)
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        vf.release()  # Release after gathering stats
        return (fps, width, height, total_frames)

    def process_frame(self, frame):
        """Process individual frames for object and lane detection."""
        # Object detection using YOLO
        model = ultralytics.YOLO("yolov8n.pt")
        results = model(frame)

        # Convert YOLOv8 results to DataFrame
        tracking_results = convert_tracking_results_to_pandas(results)

        # Draw object detection boxes
        processed_frame = self.draw_object_detection_boxes(frame, results)

        # Lane detection
        lane_frame = self.lane_detector.detect_lanes(frame)

        # Combine the lane detection and object detection results
        combined_frame = cv2.addWeighted(processed_frame, 0.8, lane_frame, 1, 0)

        return combined_frame, tracking_results

    def draw_object_detection_boxes(self, frame, results):
        """Draw bounding boxes on the frame for object detection."""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def track(self, progressbar_callback: Callable) -> tuple[pd.DataFrame, str]:
        """Perform object tracking on the video and return tracking results and the processed video path."""
        outputdir = os.getcwd()

        # Set the path for the processed video directly in mp4 format
        processed_video_path = os.path.join(outputdir, "track", self.temp_id + ".mp4")

        # Open video for processing
        vf = cv2.VideoCapture(self.video_path)
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the video writer to save directly as mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

        frame_count = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = vf.read()
            if not ret:
                break

            # Perform object detection and lane detection on each frame
            processed_frame, _ = self.process_frame(frame)

            # Write the processed frame to the output video
            out.write(processed_frame)

            # Update progress bar
            progressbar_callback(i + 1)

        # Release resources
        vf.release()
        out.release()  # Make sure the video writer is closed

        return None, processed_video_path

    def __del__(self):
        """Clean up temporary files."""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
