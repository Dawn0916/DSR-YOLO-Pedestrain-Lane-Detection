import cv2
import numpy as np
import pandas as pd

# Load object detection results from CSV
object_detection_results = pd.read_csv('combine_lane_object/road_0160_object_detection_results.csv')
lane_left = pd.read_csv('combine_lane_object/road_0160_lane_left_coordinates.csv')
lane_right = pd.read_csv('combine_lane_object/road_0160_lane_right_coordinates.csv')

# object_detection_results = pd.read_csv('combine_lane_object/road_0095_object_detection_results.csv')
# lane_left = pd.read_csv('combine_lane_object/road_0095_lane_left_coordinates.csv')
# lane_right = pd.read_csv('combine_lane_object/road_0095_lane_right_coordinates.csv')

# Load the video
video_path = 'combine_lane_object/road_0160_combined.mp4'
output_video_path = 'combine_lane_object/road_0160_combined_danger.mp4'
# video_path = 'combine_lane_object/road_0095_combined.mp4'
# output_video_path = 'combine_lane_object/road_0095_combined_danger.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def find_the_closet_x(object_center_x, lane_coords):
    # Extract the x-coordinates (first column)
    x_coords = lane_coords[:, 0]
    y_coords = lane_coords[:, 1]
    # Step 1: Check if any x-coordinates match object_center_x exactly
    exact_matches = x_coords[x_coords == object_center_x]
    x_to_compare = exact_matches
    # Get the indices of the exact matches in x_coords
    match_index = np.where(x_coords == object_center_x)[0]
    if len(match_index)>1:
        compare_index = match_index[0]
    else:
        compare_index = match_index
    y_to_compare = y_coords[compare_index]

    # if len(exact_matches) > 0:
    #     print(f"Exact matches found: {exact_matches}")
    # else:
    if len(exact_matches) <= 0:
        # Step 2: Find the closest x-coordinate to object_center_x
        closest_index = np.argmin(np.abs(x_coords - object_center_x))
        closest_value = x_coords[closest_index]
        x_to_compare = closest_value
        y_to_compare = y_coords[closest_index]
    return x_to_compare, y_to_compare
    

def is_in_lane(bbox, lane_left_coords, lane_right_coords):
    """
    Function to check if the bounding box overlaps with the lane area.
    bbox: (x_min, y_min, x_max, y_max) of the detected object.
    lane_left_coords: List of (x, y) coordinates of the left lane.
    lane_right_coords: List of (x, y) coordinates of the right lane.
    """
    x_min, y_min, x_max, y_max = bbox
    object_center_x = (x_min + x_max) // 2
    # object_center_y = (y_min + y_max) // 2
    object_center_y = y_max


    # Get the first column for left lane(x-coordinates)
    # Get the minimum and maximum of the first column
    left_x_min = np.min(lane_left_coords[:, 0])
    left_x_max = np.max(lane_left_coords[:, 0])

     # Get the second column for left lane (y-coordinates)
    # Get the minimum and maximum of the second column
    left_y_min = np.min(lane_left_coords[:, 1])
    left_y_max = np.max(lane_left_coords[:, 1])

    # Get the first column for right lane(x-coordinates)
    # Get the minimum and maximum of the first column
    right_x_min = np.min(lane_right_coords[:, 0])
    right_x_max = np.max(lane_right_coords[:, 0])

     # Get the second column for right lane (y-coordinates)
    # Get the minimum and maximum of the second column
    right_y_min = np.min(lane_right_coords[:, 1])
    right_y_max = np.max(lane_right_coords[:, 1])

    if left_x_min <= object_center_x <= right_x_max:
        if left_x_min <= object_center_x <= left_x_max:
            left_x_to_compare, left_y_to_compare = find_the_closet_x(object_center_x, lane_left_coords)
            if left_y_to_compare <= object_center_y <= left_y_max:
                return True
    
        elif left_x_max <= object_center_x <= right_x_min:
            if min(left_y_min, right_y_min) <= object_center_y <= max(left_y_max, right_y_max):
                return True
        elif right_x_min <= object_center_x <= right_x_max:
            right_x_to_compare, right_y_to_compare = find_the_closet_x(object_center_x, lane_right_coords)
            if right_y_to_compare <= object_center_y <= right_y_max:
                return True
    return False

    # # We can use simple range comparison for lane boundary, assuming lane_left/right as a polygon
    # # Check if object center lies between the left and right lane coordinates

    # if lane_left_coords[0][0] <= object_center_x <= lane_right_coords[-1][0]:
    #     return True
    # return False

# Iterate through each frame in the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get current frame object detections
    frame_objects = object_detection_results[object_detection_results['frame_no'] == frame_count]
    
    # Convert lane coordinates to numpy arrays
    lane_left_coords = lane_left[['x', 'y']].to_numpy()
    lane_right_coords = lane_right[['x', 'y']].to_numpy()

    # Draw the lanes on the frame
    lane_overlay = np.copy(frame)
    cv2.polylines(lane_overlay, [lane_left_coords], False, (0, 255, 0), 5)  # Left lane in green
    cv2.polylines(lane_overlay, [lane_right_coords], False, (0, 255, 0), 5)  # Right lane in green

    danger = False
    
    # Check each detected object in the current frame
    for index, obj in frame_objects.iterrows():
        label = obj['class_name']
        x_center = int(obj['x'])
        y_center = int(obj['y'])
        w = int(obj['w'])
        h = int(obj['h'])
        x_min = x_center - w // 2
        y_min = y_center - h // 2
        x_max = x_center + w // 2
        y_max = y_center + h // 2

        # Only check for pedestrians or motorcycles
        if label in ['person', 'motorcycle']:
            # Check if the object is in the lane
            if is_in_lane((x_min, y_min, x_max, y_max), lane_left_coords, lane_right_coords):
                danger = True
                # Debugging
                # print(f"lane_left_coords: {lane_left_coords}")
                # print(f"lane_right_coords: {lane_right_coords}")
                # print(f"lane_left_coords[0][0]: {lane_left_coords[0][0] }")
                # print(f"lane_left_coords[-1][0]: {lane_left_coords[-1][0] }")
                # print(f"lane_right_coords[0][0]: {lane_right_coords[0][0]}")
                # print(f"lane_right_coords[-1][0]: {lane_right_coords[-1][0]}")
                # print(f"padestrian position: {(x_min, y_min, x_max, y_max)}")

                # Draw bounding box for detected object
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, 'DANGER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    # If danger is detected, add the warning to the frame
    if danger:
        cv2.putText(frame, 'DANGER', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

# Release the video objects
cap.release()
out.release()

print(f'Processed video saved as {output_video_path}')
