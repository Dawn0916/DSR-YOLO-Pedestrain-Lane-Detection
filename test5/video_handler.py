import torch
import pandas as pd
from yolo_helper import convert_tracking_results_to_pandas, is_pedestrian_in_lane
import ultralytics
import cv2
from lane_detection import LaneLines

class VideoHandler:
    def __init__(self):
        # Load the YOLOv8 model
        self.model = ultralytics.YOLO("yolov8n.pt")  # Replace with the path to your model if necessary
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.lane_detector = LaneLines()  # Initialize the lane detector
        self.results_df = pd.DataFrame()

    def process_frame(self, frame):
        # Perform YOLOv8 object detection on the given frame
        results = self.model(frame, conf=0.2, iou=0.6, device=self.device)

        # Convert YOLOv8 tracking results to a DataFrame
        tracking_results = convert_tracking_results_to_pandas(results)

        # Perform lane detection on the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for lane detection
        lane_frame, left_lane, right_lane = self.lane_detector.forward(gray_frame)

        # Check for overlap between pedestrians and lanes
        danger_alarm = False
        if not tracking_results.empty:
            danger_alarm = is_pedestrian_in_lane(tracking_results, left_lane, right_lane)

        # Overlay lane lines on the processed frame
        final_frame = cv2.addWeighted(frame, 1, lane_frame, 0.7, 0)

        # Draw YOLO bounding boxes on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(final_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Return the processed frame, detection results, and danger status
        return final_frame, tracking_results, danger_alarm
