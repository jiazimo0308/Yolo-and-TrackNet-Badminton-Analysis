#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/21
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from shapely.geometry import Polygon,Point

class PoseDetector():
    def __init__(self, court_coords, mid_coords, model_path):
        self.model = YOLO(model_path)
        self.court_polygon = Polygon(court_coords)
        self.mid_coords = mid_coords.astype(np.int32)
        self.court_polygon_buffered = self.court_polygon.buffer(20)
        self.pose_data = {"P1": (0.0, 0.0), "P2": (0.0, 0.0)}

    # ======================
    # 核心：只检测，不画图
    # ======================
    def detect_pose_frame(self, frame):
        if self.court_polygon is None:
            raise Exception("请设置球场边界")

        # 每次清空数据
        self.pose_data = {"P1": (0.0, 0.0), "P2": (0.0, 0.0)}
        results = self.model(frame, verbose=False)

        if len(results) == 0:
            return []  # 返回空列表，不画图

        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        mid_p1, mid_p2 = self.mid_coords
        x1_mid, y1_mid = mid_p1
        x2_mid, y2_mid = mid_p2

        detected_poses = []  # 存储：标签 + 骨骼点

        for person_kpts in keypoints_list:
            # 双脚中心点（和你跟踪器完全一致）
            foot1 = person_kpts[15]
            foot2 = person_kpts[16]
            cx = (foot1[0] + foot2[0]) / 2
            cy_adj = (foot1[1] + foot2[1]) / 2
            # 球场内过滤
            point = Point(cx, cy_adj)
            if not self.court_polygon_buffered.contains(point):
                continue
            # P1/P2 判断
            cross = (x2_mid - x1_mid) * (cy_adj - y1_mid) - (y2_mid - y1_mid) * (cx - x1_mid)
            label = "P1" if cross < 0 else "P2"

            # 计算角度 & 伸展
            angle, stretch = self.compute_pose_metrics(person_kpts)
            self.pose_data[label] = (angle, stretch)

            # 把【标签 + 骨骼点】存起来
            detected_poses.append((label, person_kpts))

        return detected_poses

    def compute_pose_metrics(self, person_kpts):
        # 身体倾斜角度
        shoulder_mid = np.mean(person_kpts[[5, 6]], axis=0)
        hip_mid = np.mean(person_kpts[[11, 12]], axis=0)
        body_angle_rad = np.arctan2(shoulder_mid[1] - hip_mid[1], shoulder_mid[0] - hip_mid[0])
        body_angle_deg = np.degrees(body_angle_rad) % 180

        # 腿部伸展
        left_leg_len = np.linalg.norm(person_kpts[15] - person_kpts[13])
        right_leg_len = np.linalg.norm(person_kpts[16] - person_kpts[14])
        leg_stretch_ratio = (left_leg_len + right_leg_len) / 200.0

        return body_angle_deg, leg_stretch_ratio