import os
import torch
import pandas as pd
import json
from pathlib import Path
import numpy as np

from typing import Any
from tracklab.pipeline.imagelevel_module import ImageLevelModule

import logging

log = logging.getLogger(__name__)


class ConflabLoadBBox(ImageLevelModule):
    """
    Fake person detection module to skip YOLO and load the GT bounding boxes from Conflab
    """
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.id = 0
        self.annots_path = Path(cfg.annotations) / "keypoints_and_bboxes_test.json"
        log.info(f"Loading annotations from {self.annots_path}")
        with open(self.annots_path, "r") as f:
            self.annots = json.load(f)
            self.annots = self.annots["annotations"]
            self.annots = pd.DataFrame(self.annots)
        

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):        
        _detections = []
        
        for _, metadata in metadatas.iterrows():
            for bbox in (self.annots[self.annots["image_id"] == metadata.name])["bbox"]:
                _detections.append(
                    pd.Series(
                        dict(
                            image_id=metadata.name,
                            bbox_ltwh=np.array(bbox),
                            bbox_conf=1,
                            video_id=metadata.video_id,
                            category_id=1,  # `person` class in posetrack
                        ),
                        name=self.id,
                    )
                )
                self.id += 1

        return _detections
