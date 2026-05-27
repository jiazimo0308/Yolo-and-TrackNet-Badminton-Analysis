#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jiazimo
# @Time : 2026/5/26 速度版异常过滤+自动reset
import torch
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque
from Track_Model.V3.model import TrackNetV2 as TrackNet


class ShuttleDetector:
    def __init__(self, weights_path, imgsize=(288, 512), max_speed=1500):
        # 模型参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = TrackNet(in_dim=9, out_dim=3).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.imgsize = imgsize
        self.frame_queue = []#存储连续3帧输入
        #速度判断
        self.max_speed = max_speed#最大允许速度（像素/秒）
        # 队列
        self.two_queue = deque(maxlen=2)#用于计算帧间速度
        self.point_queue = deque(maxlen=4) #用于拟合
        # 连续拟合保护
        self.consecutive_fit_count = 0
        self.max_consecutive_fits = 3
        #轨迹保存
        self.trajectory = deque(maxlen=120)

    def find_shuttle(self, bin_img):
        """从二值图中找羽毛球轮廓，返回中心坐标"""
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if 5 < cv2.contourArea(c) < 120:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    return (cx, cy)
        return None

    def detect(self, frame,fps):
        self.fps=fps
        self.frame_time = 1.0 / fps#每帧时间（秒）
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.imgsize[1], self.imgsize[0]))
        inp = torch.from_numpy(resized.transpose(2, 0, 1)).float().div(255.0)
        # 维护3帧输入队列
        self.frame_queue.append(inp)
        if len(self.frame_queue) > 3:
            self.frame_queue.pop(0)
        if len(self.frame_queue) < 3:
            return None,list(self.trajectory)
        # 模型推理
        imgs_torch = torch.cat(self.frame_queue, dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(imgs_torch).cpu().numpy()[0]
        pred = pred.squeeze()
        max_conf = np.max(pred[2])
        if max_conf > 0.8:
            w1, w2, w3 = 0.1, 0.15, 0.75  # 实时帧清晰，高权重
        elif max_conf < 0.5:
            w1, w2, w3 = 0.25, 0.35, 0.4  # 实时帧模糊，依赖历史
        else:
            w1, w2, w3 = 0.15, 0.25, 0.6  # 默认权重
        pred_fused = pred[0] * w1 + pred[1] * w2 + pred[2] * w3
        pred_bin = (pred_fused > 0.45).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
        # 找羽毛球坐标并还原到原图尺寸
        shuttle_pos = self.find_shuttle(pred_bin)
        if shuttle_pos:
            x = shuttle_pos[0] * w / self.imgsize[1]
            y = shuttle_pos[1] * h / self.imgsize[0]
            final_pos = self.SoFar(x, y)
            self.trajectory.append(final_pos)
            return final_pos, list(self.trajectory)
        # 检测不到羽毛球直接reset
        self.reset()
        return None,list(self.trajectory)

    def fit_3_to_4(self):
        if len(self.point_queue) < 3:
            return None
        points = np.array(list(self.point_queue)[-3:], dtype=np.float32)
        t = np.arange(3)
        try:
            x_coeffs = np.polyfit(t, points[:, 0], 2)
            y_coeffs = np.polyfit(t, points[:, 1], 2)
            t_next = 3
            pred_x = x_coeffs[0] * t_next ** 2 + x_coeffs[1] * t_next + x_coeffs[2]
            pred_y = y_coeffs[0] * t_next ** 2 + y_coeffs[1] * t_next + y_coeffs[2]
            return (int(pred_x), int(pred_y))
        except (np.linalg.LinAlgError, ValueError):
            return None

    def SoFar(self, x, y):
        curr_raw_point = (x, y)
        if len(self.two_queue) < 2:
            self.two_queue.append(curr_raw_point)
            self.point_queue.append(curr_raw_point)
            self.consecutive_fit_count = 0
            return (int(curr_raw_point[0]), int(curr_raw_point[1]))

        prev_point = self.two_queue[-1]
        dx = curr_raw_point[0] - prev_point[0]
        dy = curr_raw_point[1] - prev_point[1]
        frame_distance = np.sqrt(dx ** 2 + dy ** 2)
        current_speed = frame_distance / self.frame_time

        #速度正常使用实际点
        if current_speed < self.max_speed:
            self.two_queue.append(curr_raw_point)
            self.point_queue.append(curr_raw_point)
            self.consecutive_fit_count = 0
            return (int(curr_raw_point[0]), int(curr_raw_point[1]))
        #速度异常尝试拟合
        else:
            # 连续拟合超过3次，直接reset
            if self.consecutive_fit_count >= self.max_consecutive_fits:
                self.reset()
                return (int(curr_raw_point[0]), int(curr_raw_point[1]))

            fitted_point = self.fit_3_to_4()
            # 拟合不上直接调用reset
            if fitted_point is None:
                self.reset()
                return (int(curr_raw_point[0]), int(curr_raw_point[1]))

            # 拟合成功，使用拟合点
            self.two_queue.append(fitted_point)
            self.point_queue.append(fitted_point)
            self.consecutive_fit_count += 1
            return fitted_point

    def reset(self):
        """重置所有队列和状态"""
        self.frame_queue.clear()
        self.two_queue.clear()
        self.point_queue.clear()
        self.consecutive_fit_count = 0
        self.trajectory.clear()


if __name__ == '__main__':
    # 配置路径
    WEIGHTS_PATH = "/Track_Model/V3/model_best.pt"
    VIDEO_PATH = "/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/videos/Video Project.mp4"

    # 获取视频实际帧率（自动适配不同视频）
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # 自动获取失败时用默认值
    print(f"检测到视频帧率: {video_fps:.1f} fps")
    # 初始化检测器（传入实际帧率）
    detector = ShuttleDetector(WEIGHTS_PATH,max_speed=1500,fps=video_fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_point = None

    print("🏸 速度版TrackNet羽毛球追踪已启动 | 按Q退出")
    for _ in tqdm(range(total_frames), desc="处理进度"):
        ret, frame = cap.read()
        if not ret:
            break

        current_point = detector.detect(frame)

        if current_point is not None:
            if last_point is not None:
                cv2.line(frame, last_point, current_point, (0, 0, 255), 2)
            cv2.circle(frame, current_point, 6, (0, 0, 255), -1)
            last_point = current_point
        else:
            last_point = None

        cv2.imshow("速度版TrackNet追踪", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 处理完成")