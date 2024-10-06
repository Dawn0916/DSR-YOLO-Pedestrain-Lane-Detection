import cv2
import os

# Paths for input object detection video and lane detection video
object_detection_video_path = "combine_lane_object/road_0160_object_detection.mp4"
lane_detection_video_path = "combine_lane_object/road_0160_lane_detection.mp4"
# object_detection_video_path = "combine_lane_object/road_0095_object_detection.mp4"
# lane_detection_video_path = "combine_lane_object/road_0095_lane_detection.mp4"

# Path to save the final combined output video
combined_output_video_path = "combine_lane_object/road_0160_combined.mp4"
# combined_output_video_path = "combine_lane_object/road_0095_combined.mp4"

# Open both videos
object_cap = cv2.VideoCapture(object_detection_video_path)
lane_cap = cv2.VideoCapture(lane_detection_video_path)

# Get properties from the first video (assuming both have same dimensions and fps)
fps = int(object_cap.get(cv2.CAP_PROP_FPS))
frame_width = int(object_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(object_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(combined_output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read frames from both videos
    ret_obj, object_frame = object_cap.read()
    ret_lane, lane_frame = lane_cap.read()

    if not ret_obj or not ret_lane:
        # If either video ends, break the loop
        break

    # Combine frames by overlaying (you can use cv2.addWeighted or any other method)
    # combined_frame = object_frame 
    combined_frame = cv2.addWeighted(lane_frame, 0.5, object_frame, 0.5, 0)

    # Write the combined frame to the output video
    out.write(combined_frame)

# Release video resources
object_cap.release()
lane_cap.release()
out.release()

print("Combined video saved at:", combined_output_video_path)
