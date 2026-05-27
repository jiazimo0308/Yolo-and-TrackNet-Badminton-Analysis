#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/22
import cv2
import numpy as np
import pandas as pd

class Mapping_changdi():#轨迹点的坐标映射并通过小地图展现
    def __init__(self,court_coords):
        self.court_coords=court_coords
        self.real_court=np.array([[0, 0],[7.1, 0],[7.1, 15],[0, 15]], dtype=np.float32)
        self.H = None  # 映射矩阵
        self.trajectories = None  # 外部传入轨迹

    def trajectories_in(self, trajectories):
        '''从外部传入轨迹数据'''
        self.trajectories = trajectories

    def CaH(self):#求映射矩阵
        pixel_court = self.court_coords.reshape(4, 2).astype(np.float32)
        self.H, _ = cv2.findHomography(pixel_court, self.real_court)

    def track_mapping(self):
        self.CaH()
        mapped_trajectories = {}
        for label, points in self.trajectories.items():
            mapped_points = []
            for (frame, x, y) in points:
                real_x, real_y = self._map(x, y)
                mapped_points.append((frame, real_x, real_y))
            mapped_trajectories[label] = mapped_points
        return mapped_trajectories

    def _map(self, x, y):
        if self.H is None:
            self.CaH()
        pt = np.array([x, y, 1.0], dtype=np.float32).reshape(3, 1)
        res = self.H @ pt
        res /= res[2]  # 归一化
        real_x = round(float(res[0][0]), 2)
        real_y = round(float(res[1][0]), 2)
        return real_x, real_y

    def map_single_point(self, x, y):
        """ 单个点映射球员都用这个 """
        return self._map(x, y)

    def map_badminton_with_height(self, x, y):
        # 取出你的场地四个角
        c = self.court_coords.reshape(4,2)
        tl, tr, br, bl = c

        # 计算场地中轴线（透视消失线）
        cx1 = (tl[0] + tr[0])/2
        cy1 = (tl[1] + tr[1])/2
        cx2 = (bl[0] + br[0])/2
        cy2 = (bl[1] + br[1])/2

        # 计算从消失线指向球的向量（高度带来的偏移方向）
        dx_line = cx2 - cx1
        dy_line = cy2 - cy1

        # 核心：
        # 羽毛球越高，越靠近消失点
        # 把它沿着透视方向推回地板
        k = 0.28   # 透视强度（固定值，所有比赛通用）
        x_floor = x - dx_line * ((y - cy1) / (cy2 - cy1)) * k
        y_floor = y - dy_line * ((y - cy1) / (cy2 - cy1)) * k

        # 映射到小地图（和球员完全一致）
        return self._map(x_floor, y_floor)






