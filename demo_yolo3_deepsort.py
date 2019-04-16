import os
import cv2
import numpy as np

from YOLO3 import YOLO3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.batch_size = 36
        self.yolo3 = YOLO3(
            "YOLO3/cfg/yolo_v3.cfg",
            "YOLO3/yolov3.weights",
            "YOLO3/cfg/coco.names",
            is_xywh=True,
        )
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(
                "demo.avi", fourcc, 2.5, (self.im_width, self.im_height)
            )
        return self.vdo.isOpened()

    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        ims, ori_ims = [], []
        start = time.time()
        while self.vdo.grab():
            _, ori_im = self.vdo.retrieve()
            ori_ims.append(ori_im)
            ims.append(ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)])
            if len(ims) == self.batch_size:
                batch_bbox_xywh, batch_cls_conf, batch_cls_ids = self.yolo3(ims)
                assert len(batch_bbox_xywh) == self.batch_size
                for i, (bbox_xywh, cls_conf, cls_ids) in enumerate(
                    zip(batch_bbox_xywh, batch_cls_conf, batch_cls_ids)
                ):

                    if bbox_xywh is not None:
                        mask = cls_ids == 0
                        bbox_xywh = bbox_xywh[mask]
                        bbox_xywh[:, 3] *= 1.2
                        cls_conf = cls_conf[mask]
                        outputs = self.deepsort.update(bbox_xywh, cls_conf, ims[i])
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            ori_ims[i] = draw_bboxes(
                                ori_ims[i], bbox_xyxy, identities, offset=(xmin, ymin)
                            )
                    if self.write_video:
                        self.output.write(ori_ims[i])
                ims, ori_ims = [], []

                end = time.time()
                print(
                    "time: {}s, fps: {}".format(
                        end - start, self.batch_size / (end - start)
                    )
                )
                start = time.time()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    else:
        det = Detector()
        det.open(sys.argv[1])
        det.detect()
