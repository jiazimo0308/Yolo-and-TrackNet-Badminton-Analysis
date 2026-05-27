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

class PlayerDetector():
    def __init__(self,court_coords,mid_coords,model_path):
        self.model = YOLO(model_path)
        self.model.conf = 0.4
        self.offset_ratio = 0.4
        self.track_offset_ratio = 0.25
        self.court_polygon = Polygon(court_coords)
        #
        self.tracker_cfg='bytetrack.yaml'
        self.player_map = {}
        self.trajectories = defaultdict(list)
        self.frame_height = None
        #跟踪功能必需的辅助属性
        self.court_coords = court_coords.astype(np.int32)
        self.mid_coords=mid_coords.astype(np.int32)
        self.court_polygon_buffered = self.court_polygon.buffer(20)
        self.min_y_court_top = np.min(self.court_coords[:, 1])

    def _is_player_in_court(self, x1, y1, x2, y2):
        '''判断球员是否在场地内'''
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        box_h = int(y2 - y1)
        if box_h <= 0:
            return False, cx, cy
        cy_adj = int(cy + self.offset_ratio * box_h)
        point = Point(cx, cy_adj)
        return self.court_polygon.contains(point), cx, cy_adj

    def detect(self, frame):
        '''监测场地中的所以球员'''
        if self.court_polygon is None:
            raise Exception("请设置球场边界")
        results = self.model(frame, verbose=False)  # verbose=False关闭YOLO的冗余输出
        valid_players = []
        for r in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, class_id = r[:6]
            # 只保留"人"的检测结果
            if int(class_id) != 0:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            is_inside, cx, cy_adj = self._is_player_in_court(x1, y1, x2, y2)
            if is_inside:
                valid_players.append(((x1, y1, x2, y2), (cx, cy_adj)))
        return valid_players

    def track_frame(self, frame, frame_idx):
        '''跟踪单帧图像（用球场中线划分身份）'''
        if self.court_polygon is None:
            raise Exception("请设置球场边界")
        results = self.model.track(frame,persist=True,tracker=self.tracker_cfg,verbose=False,iou=0.5)
        valid_players = []
        if len(results[0].boxes) == 0:
            return valid_players
        mid_p1 = self.mid_coords[0]
        mid_p2 = self.mid_coords[1]
        x1_mid, y1_mid = mid_p1
        x2_mid, y2_mid = mid_p2
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, track_id, score, class_id = box[:7]
            if int(class_id) != 0:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            box_h = int(y2 - y1)
            cy_adj = int(cy + self.track_offset_ratio * box_h)
            point_to_check = Point(cx, cy_adj)
            if not self.court_polygon_buffered.contains(point_to_check):
                continue
            if y2 < self.min_y_court_top - 20:
                continue
            if track_id not in self.player_map:
                # 用球场中线划分身份
                cross = (x2_mid - x1_mid) * (cy_adj - y1_mid) - (y2_mid - y1_mid) * (cx - x1_mid)
                if cross < 0:
                    label = "P1"
                else:
                    label = "P2"
                self.player_map[track_id] = label
                print(f"分配身份：跟踪ID {int(track_id)} → {label}")
            label = self.player_map[track_id]
            self.trajectories[label].append((frame_idx, cx, cy_adj))
            valid_players.append(((x1, y1, x2, y2), (cx, cy_adj), label))
        return valid_players

    def save_trajectories(self, output_dir="./results"):
        """保存轨迹到CSV文件"""
        os.makedirs(output_dir, exist_ok=True)
        for label, data in self.trajectories.items():
            df = pd.DataFrame(data, columns=["frame", "x", "y"])
            out_path = f"{output_dir}/{label}_trajectory.csv"
            df.to_csv(out_path, index=False)
            print(f"已保存 {label} 轨迹 → {out_path}")

    def reset_tracker(self):
        """重置跟踪器状态"""
        self.player_map.clear()
        self.trajectories.clear()
        self.frame_height = None
        print("跟踪器已重置")






