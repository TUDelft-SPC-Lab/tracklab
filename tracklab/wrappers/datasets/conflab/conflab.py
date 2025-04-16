import logging
import os
import numpy as np
import pandas as pd
import json

from pathlib import Path
from tqdm import tqdm
from tracklab.datastruct import TrackingDataset, TrackingSet

log = logging.getLogger(__name__)


class ConflabMOT(TrackingDataset):
    def __init__(self,
                 dataset_path: str,
                 nvid: int = -1,
                 vids_dict: list = None,
                 *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), f"'{self.dataset_path}' directory does not exist. Please check the path or download the dataset following the instructions here: https://github.com/SoccerNet/sn-tracking"

        log.info(f"Loading Conflab MOT dataset from {self.dataset_path} ...")
        # train_set = load_set(self.dataset_path / "train", nvid, vids_dict["train"])
        train_set = load_set(self.dataset_path / "images_test",self.dataset_path /"keypoints_and_bboxes_test.json", nvid, vids_dict["train"])
        # test_set = load_set(self.dataset_path / "test", nvid, vids_dict["val"])
        test_set = train_set
        
        sets = {
            "train": train_set,
            "test": test_set,
            "val": test_set,
        }

        super().__init__(dataset_path, sets, nvid=-1, vids_dict=None, *args, **kwargs)


# def read_ini_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     return dict(line.strip().split('=') for line in lines[1:])


def read_motchallenge_formatted_file(file_path):
    # columns = ['image_id', 'track_id', 'left', 'top', 'width', 'height', 'bbox_conf', 'class', 'visibility', 'unused']
    # df = pd.read_csv(file_path, header=None, names=columns)
    # df['bbox_ltwh'] = df.apply(lambda row: np.array([row['left'], row['top'], row['width'], row['height']]), axis=1)
    # return df[['image_id', 'track_id', 'bbox_ltwh', 'bbox_conf', 'class', 'visibility']]
    with open(file_path, 'r') as file:
        raw_data = json.load(file)
    # df = pd.read_json(file_path)
    image_id =  []
    track_id = []
    bbox_ltwh = []
    bbox_conf = []
    class_annot = []
    visibility = []
    unused = []

    for data_entry in tqdm(raw_data["annotations"], desc="Processing annotations"):
        image_id.append(data_entry["image_id"])
        track_id.append(1) # TODO FIX THIS ONE, THIS IS THE PERSON ID
        bbox_ltwh.append(data_entry["bbox"])
        bbox_conf.append(1)
        class_annot.append(-1)
        visibility.append(-1)
        unused.append(-1)


    df = pd.DataFrame({
        'image_id': image_id,
        'track_id': track_id,
        'bbox_ltwh': bbox_ltwh,
        'bbox_conf': bbox_conf,
        'class': class_annot,
        'visibility': visibility,
        "unused": unused,
    })
    return df


def load_set(img_folder_path, annotations_path, nvid=-1, vids_filter_set=None):
    video_metadatas_list = []
    image_metadata_list = []
    detections_list = []
    categories_list = []
    
    image_counter = 0
    person_counter = 0

    # Read ground truth detections
    detections_df = read_motchallenge_formatted_file(annotations_path)
    detections_df['person_id'] = detections_df['track_id']
    detections_df['video_id'] = 1
    detections_df['visibility'] = 1
    detections_list.append(detections_df)

    # Append video metadata
    nframes = len(detections_df['image_id'].unique())
    video_metadata = {
        'id': len(video_metadatas_list) + 1,
        'name': '',
        'nframes': nframes,
        'frame_rate': 60,
        'seq_length': nframes,
        'im_width': 960,
        'im_height': 540,
        'game_id': 0,
        'action_position': 0,
        'action_class': '',
        'visibility': '',
        'clip_start': 0,
        # 'game_time_start': 0,
        # # Remove the half period index
        # 'game_time_stop': gameinfo_data.get('gameTimeStop', '').split(' - ')[1],  # Remove the half period index
        # 'clip_stop': int(gameinfo_data.get('clipStop', 0)),
        # 'num_tracklets': int(gameinfo_data.get('num_tracklets', 0)),
        # 'half_period_start': int(gameinfo_data.get('gameTimeStart', '').split(' - ')[0]),
        # # Add the half period start column
        # 'half_period_stop': int(gameinfo_data.get('gameTimeStop', '').split(' - ')[0]),
        # # Add the half period stop column
    }

    # Append video metadata
    video_metadatas_list.append(video_metadata)

    # Append image metadata
    img_metadata_df = pd.DataFrame({
        'frame': [i for i in range(0, nframes)],
        'id': [image_counter + i for i in range(0, nframes)],
        'video_id': len(video_metadatas_list),
        'file_path': [os.path.join(img_folder_path, f'{i:09d}.jpg') for i in
                        range(1, nframes + 1)],

    })
    image_counter += nframes
    person_counter += len(detections_df['track_id'].unique())
    image_metadata_list.append(img_metadata_df)

    categories_list = [{'id': i + 1, 'name': category, 'supercategory': 'person'} for i, category in
                       enumerate(sorted(set(categories_list)))]

    # Assign the categories to the video metadata  # TODO at dataset level?
    for video_metadata in video_metadatas_list:
        video_metadata['categories'] = categories_list

    # Concatenate dataframes
    video_metadata = pd.DataFrame(video_metadatas_list)  # FIXME video_metadata ??
    image_metadata = pd.concat(image_metadata_list, ignore_index=True)
    detections = pd.concat(detections_list, ignore_index=True)

    # Use video_id, image_id, track_id as unique id
    detections = detections.sort_values(by=['video_id', 'image_id', 'track_id'], ascending=[True, True, True])
    detections['id'] = detections['video_id'].astype(str) + "_" + detections['image_id'].astype(str) + "_" + detections[
        'track_id'].astype(str)

    # Add category id to detections
    category_to_id = {category['name']: category['id'] for category in categories_list}
    # detections['category_id'] = detections['category'].apply(lambda x: category_to_id[x])

    detections.set_index("id", drop=False, inplace=True)
    image_metadata.set_index("id", drop=False, inplace=True)
    video_metadata.set_index("id", drop=False, inplace=True)

    # Add is_labeled column to image_metadata
    image_metadata['is_labeled'] = True

    # Reorder columns in dataframes
    video_metadata_columns = ['name', 'nframes', 'frame_rate', 'seq_length', 'im_width', 'im_height', 'game_id', 'action_position',
                   'action_class', 'visibility', 'clip_start', 
                #    'game_time_start', 'clip_stop', 'game_time_stop',
                #    'num_tracklets',
                #    'half_period_start', 'half_period_stop', 
                   'categories']
    video_metadata_columns.extend(set(video_metadata.columns) - set(video_metadata_columns))
    video_metadata = video_metadata[video_metadata_columns]
    image_metadata_columns = ['video_id', 'frame', 'file_path', 'is_labeled']
    image_metadata_columns.extend(set(image_metadata.columns) - set(image_metadata_columns))
    image_metadata = image_metadata[image_metadata_columns]
    detections_column_ordered = ['image_id', 'video_id', 'track_id', 'person_id', 'bbox_ltwh', 'bbox_conf', 'class', 'visibility']
    detections_column_ordered.extend(set(detections.columns) - set(detections_column_ordered))
    detections = detections[detections_column_ordered]

    return TrackingSet(
        video_metadata,
        image_metadata,
        detections,
    )
