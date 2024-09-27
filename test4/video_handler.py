import torch
import pandas as pd
from yolo_helper import convert_tracking_results_to_pandas
import ultralytics
import cv2
# from lane_detection import LaneLines  # Import the lane detection class

class VideoHandler:
    def __init__(self):
        # Load the YOLOv8 model
        self.model = ultralytics.YOLO("yolov8n.pt")  # Replace with the path to your model if necessary
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        # self.lane_detector = LaneLines()  # Initialize the lane detector
        self.results_df = pd.DataFrame()

    def process_frame(self, frame):
        # Perform YOLOv8 object detection on the given frame
        results = self.model(frame, conf=0.2, iou=0.6, device=self.device)
        
        # Convert YOLOv8 tracking results to a DataFrame
        tracking_results = convert_tracking_results_to_pandas(results)

        # Optional: Draw bounding boxes on the frame (for visualization)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Draw bounding boxes and class labels
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # # Perform lane detection on the frame
        # frame = self.lane_detector.forward(frame)

        # Return the processed frame with both YOLO object detection and lane detection
        return frame, tracking_results
