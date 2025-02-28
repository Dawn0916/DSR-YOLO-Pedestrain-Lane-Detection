import ultralytics
#import os
#import torch
#from video_helper import get_video_properties
import pandas as pd
import numpy as np

frames = 0

def make_callback_adapter_with_counter(event_name, callback):
    """
    Convert the callback function with 2 params to a callback format required by YOLO.
    Args:
        event_name: str: YOLO pipeline event name
        callback: Callable(str, int): a callback function accepting an 2 params: event_name and counter
    Return:
        A callback in the format required by YOLO.
    """
    event_counter = 0

    def yolo_callback(component):
        nonlocal event_counter
        event_counter += 1
        callback(event_name, event_counter)

    return yolo_callback

def _convert_single_tracking_result(frame_no, boxes_result:ultralytics.engine.results.Boxes):
    """ 
    This function takes in a frame number and bounding box results (from YOLOv8) 
    and converts them into a Pandas DataFrame
    """
    box = boxes_result.boxes # sic!   # Extract Bounding Box Information
    int_vectorized = np.vectorize(np.int_, otypes=[int])  # Vectorized Integer Conversion
    if box is not None: # Check if Bounding Box Exists
        # Extract Object Information
        class_ids = int_vectorized(box.cls.cpu().numpy()) 
        observation_count = len(class_ids)
        # Map Class IDs to Class Names
        class_id_to_name = lambda id: boxes_result.names[int(id)]
        class_names = list(map(class_id_to_name, class_ids))
        # Extract Object IDs (if available)
        ids = int_vectorized(box.id.cpu()) if box.id is not None else np.zeros(shape=observation_count, dtype='int')
        # Extract Bounding Box Coordinates (x, y, width, height)
        xywh = box.xywh.cpu()
        xs = xywh[:, 0]
        ys = xywh[:, 1]
        ws = xywh[:, 2]
        hs = xywh[:, 3]
        # Create Frame Numbers
        frame_nos = np.repeat(a=frame_no, repeats=observation_count)
        # Create a Dictionary of the Data
        data = dict(frame_no=frame_nos, class_id=class_ids, class_name=class_names, id=ids, x=xs, y=ys, w=ws, h=hs)
        # Convert the Dictionary to a DataFrame
        df = pd.DataFrame(data=data)
        return df
    else:  # Handle the Case Where No Boxes Are Detected: returns an empty DataFrame with the appropriate column names.
        return pd.DataFrame(columns=['frame_no','class_id', 'class_name', 'id', 'x', 'y', 'w', 'h'])

def convert_tracking_results_to_pandas(tracking_results):
    """
    Convert YOLOv8 tracking output to a Pandas DataFrame.
    The DataFrame contains the following columns:
        - frame_no:int - frame number
        - class_id:int - class identifier
        - class_name:str - class name of the tracked object
        - id:int - identifier of the tracked object
        - x:int - coordinates of the bounding boxes
        - y:int
        - w:int
        - h::int
    """
    dfs = [] # Will contain 1 data frame per video frame
    for i, tr in enumerate(tracking_results):
        df = _convert_single_tracking_result(i, tr)
        dfs.append(df)

    return pd.concat(dfs)