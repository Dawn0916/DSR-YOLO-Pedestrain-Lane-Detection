import pandas as pd
import numpy as np

def _convert_single_tracking_result(frame_no, boxes_result):
    box = boxes_result.boxes
    int_vectorized = np.vectorize(np.int_, otypes=[int])
    if box is not None:
        class_ids = int_vectorized(box.cls.cpu().numpy())
        observation_count = len(class_ids)
        class_id_to_name = lambda id: boxes_result.names[int(id)]
        class_names = list(map(class_id_to_name, class_ids))
        ids = int_vectorized(box.id.cpu()) if box.id is not None else np.zeros(shape=observation_count, dtype='int')
        xywh = box.xywh.cpu()
        xs = xywh[:, 0]
        ys = xywh[:, 1]
        ws = xywh[:, 2]
        hs = xywh[:, 3]
        frame_nos = np.repeat(a=frame_no, repeats=observation_count)
        data = dict(frame_no=frame_nos, class_id=class_ids, class_name=class_names, id=ids, x=xs, y=ys, w=ws, h=hs)
        df = pd.DataFrame(data=data)
        return df
    else:
        return pd.DataFrame(columns=['frame_no', 'class_id', 'class_name', 'id', 'x', 'y', 'w', 'h'])

def convert_tracking_results_to_pandas(tracking_results):
    dfs = []
    for i, tr in enumerate(tracking_results):
        df = _convert_single_tracking_result(i, tr)
        dfs.append(df)
    return pd.concat(dfs)

def is_pedestrian_in_lane(tracking_results, left_lane, right_lane):
    """Check if any pedestrian is inside the lane area defined by left_lane and right_lane."""
    pedestrians = tracking_results[tracking_results['class_name'] == 'person']

    if pedestrians.empty:
        return False

    for _, pedestrian in pedestrians.iterrows():
        # Calculate pedestrian bounding box center
        pedestrian_x_center = pedestrian['x'] + pedestrian['w'] / 2

        # Check if the pedestrian's center lies between the lane lines
        if left_lane <= pedestrian_x_center <= right_lane:
            return True

    return False
