#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/21
import os
import cv2
import numpy as np
import pandas as pd

class Speed():
    def __init__(self,court_coords, fps):
        self.court_coords=court_coords
        self.fps = fps
        self.trajectories = {}
        self.speed_data = {}

    def set_trajectories(self, trajectories):
        '''从外部传入轨迹数据'''
        self.trajectories = trajectories

    def calculate_H(self):
        real_court = np.array([[0, 0],[7.1, 0],[7.1, 15],[0, 15]], dtype=np.float32)
        pixel_court = self.court_coords.astype(np.float32)
        H, _ = cv2.findHomography(pixel_court, real_court)
        return H

    def calculate_speed(self):
        """自动计算所有球员的距离、速度"""
        H = self.calculate_H()
        speed_data={}
        for label, points in self.trajectories.items():
            if len(points) < 2:
                self.speed_data[label] = []
                continue
            df = pd.DataFrame(points, columns=["frame", "x", "y"])
            x = df["x"].values
            y = df["y"].values
            ones = np.ones_like(x)
            pixel_pts = np.vstack([x, y, ones]).T
            real_pts = H @ pixel_pts.T
            real_pts = real_pts.T
            real_pts /= real_pts[:, [2]]
            x_real = real_pts[:, 0]
            y_real = real_pts[:, 1]
            dx_real = np.diff(x_real)
            dy_real = np.diff(y_real)
            dist_m = np.sqrt(dx_real ** 2 + dy_real ** 2)
            speed_mps = dist_m * self.fps
            speed_mps = np.convolve(speed_mps, np.ones(3) / 3, mode='same')
            cumulative_dist_m = np.cumsum(dist_m)#累计距离
            frames = df["frame"].values[1:]
            speed_list = []
            for f, s, d,total_d in zip(frames, speed_mps, dist_m,cumulative_dist_m):
                speed_list.append({
                    "frame": int(f),
                    "speed_mps": round(float(s), 2),
                    "dist_m": round(float(d), 2),
                    "total_dist_m": round(float(total_d), 2)
                })
            self.speed_data[label] = speed_list

    def save_speed_data(self, output_dir="./results"):
        """保存速度文件"""
        os.makedirs(output_dir, exist_ok=True)
        self.calculate_speed()
        for label, data in self.speed_data.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(f"{output_dir}/{label}_speed_data.csv", index=False)
        print("速度&距离已保存")