import torch
import pandas as pd
import cv2
import numpy as np
from yolo_helper import convert_tracking_results_to_pandas
from lane_detection import LaneDetector
import ultralytics
import streamlit as st

class VideoHandler:
    def __init__(self):
        # Load the YOLOv8 model
        self.model = ultralytics.YOLO("yolov8n.pt")  # Replace with the path to your model if necessary
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.lane_detector = LaneDetector()  # Initialize lane detection
        self.results_df = pd.DataFrame()

    def process_frame(self, frame):
        # Perform YOLOv8 object detection on the given frame
        results = self.model(frame, conf=0.2, iou=0.6, device=self.device)
        
        # Convert YOLOv8 tracking results to a DataFrame
        tracking_results = convert_tracking_results_to_pandas(results)
        
        # Optional: Draw bounding boxes on the frame (for visualization)
        processed_frame = self.draw_object_detection_boxes(frame, results)
        
        # Perform lane detection and overlay lanes on the frame
        lane_frame = self.lane_detector.detect_lanes(frame)
        combined_frame = cv2.addWeighted(processed_frame, 0.8, lane_frame, 1, 0)
        
        # Check if any pedestrians are in the lane and trigger an alarm
        self.check_pedestrian_in_lane(tracking_results, lane_frame)

        return combined_frame, tracking_results

    def draw_object_detection_boxes(self, frame, results):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Draw bounding boxes and class labels
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def check_pedestrian_in_lane(self, tracking_results, lane_img):
        # Filter tracking results to check for pedestrians
        pedestrians = tracking_results[tracking_results['class_name'] == 'person']

        # Iterate over pedestrian bounding boxes and check if they overlap with lanes
        for _, pedestrian in pedestrians.iterrows():
            x, y, w, h = pedestrian['x'], pedestrian['y'], pedestrian['w'], pedestrian['h']
            pedestrian_region = lane_img[y:y+h, x:x+w]
            if np.any(pedestrian_region == 255):  # Assuming white pixels represent lanes
                st.warning("Pedestrian detected in lane! Please be cautious.")
