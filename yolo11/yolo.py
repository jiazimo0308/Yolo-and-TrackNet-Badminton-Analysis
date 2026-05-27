#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/26
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO


class YOLOShuttleDetector:
    """
    独立的YOLO羽毛球检测器（带完整调试信息）
    """
    def __init__(self, weights_path, conf_threshold=0.3, device=None):
        """
        初始化YOLO检测器（默认降低置信度到0.3，先看有没有结果）
        """
        # 第一步：检查权重文件是否存在
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        print(f"找到权重文件: {weights_path}")

        self.conf_threshold = conf_threshold
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用推理设备: {self.device}")

        # 加载YOLO模型
        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            print(f"YOLO模型加载成功")
            print(f"模型支持的所有类别: {list(self.model.names.values())}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

        # 支持的羽毛球类别名称（自动转为小写）
        self.valid_classes = {"shuttle", "shuttlecock", "ball"}
        print(f"我们要检测的类别: {self.valid_classes}")

    def detect(self, frame, verbose=True):
        """
        检测单帧图像中的羽毛球（默认开启详细输出）
        """
        if frame is None:
            print("输入帧为空")
            return None

        # 强制设置verbose=True，看YOLO的原始输出
        results = self.model(frame, verbose=verbose, conf=self.conf_threshold)
        return self._parse_results(results)

    def _parse_results(self, results):
        """
        解析YOLO推理结果，打印所有检测到的框
        """
        yolo_boxes = results[0].boxes.data.cpu().numpy()
        print(f"\n本帧共检测到 {len(yolo_boxes)} 个目标")

        best_box = None
        max_conf = self.conf_threshold

        for i, box in enumerate(yolo_boxes):
            conf = float(box[4])
            cls_id = int(box[5])
            class_name = results[0].names[cls_id].lower()

            print(f"  目标{i+1}: 类别={class_name}, 置信度={conf:.3f}, 坐标={box[:4]}")

            # 只保留有效类别且置信度足够高的结果
            if conf >= self.conf_threshold and class_name in self.valid_classes:
                if best_box is None or conf > max_conf:
                    max_conf = conf
                    best_box = box
                    print(f"找到符合条件的羽毛球，置信度={conf:.3f}")

        if best_box is not None:
            x1, y1, x2, y2 = best_box[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"最终羽毛球坐标: ({center_x:.1f}, {center_y:.1f})")
            return (center_x, center_y)
        else:
            print("本帧未检测到符合条件的羽毛球")
            return None

    def draw_detection(self, frame, shuttle_xy, color=(0, 0, 255), radius=12, thickness=-1):
        """
        在图像上绘制检测结果（改用红色，更显眼；加大半径）
        """
        if shuttle_xy is not None:
            x, y = map(int, shuttle_xy)
            # 画一个大的实心圆+外圈，确保能看到
            cv2.circle(frame, (x, y), radius, color, thickness)
            cv2.circle(frame, (x, y), radius+3, (255,255,255), 2)
            # 显示置信度
            cv2.putText(frame, f"Shuttle", (x-30, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame


# 单独测试入口
if __name__ == '__main__':
    # 配置参数（请确认这些路径绝对正确！）
    YOLO_WEIGHT = "/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/weights/best.pt"
    VIDEO_PATH = "/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/videos/Video Project.mp4"

    # 第一步：检查视频文件是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"视频文件不存在: {VIDEO_PATH}")
        exit(1)
    print(f"找到视频文件: {VIDEO_PATH}")

    # 初始化检测器（先把置信度降到0.3，确保能看到结果）
    try:
        detector = YOLOShuttleDetector(YOLO_WEIGHT, conf_threshold=0.3)
    except Exception as e:
        print(f"检测器初始化失败: {str(e)}")
        exit(1)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频文件")
        exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频信息: 总帧数={total_frames}, FPS={fps}")

    print("\nYOLO羽毛球检测调试模式已启动，按Q退出")
    print("="*60)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n视频播放完毕")
            break

        frame_count += 1
        print(f"\n--- 第 {frame_count}/{total_frames} 帧 ---")

        # 检测羽毛球
        shuttle_xy = detector.detect(frame, verbose=True)

        # 绘制结果
        frame = detector.draw_detection(frame, shuttle_xy)

        # 显示（确保窗口大小合适）
        cv2.namedWindow("YOLO Shuttle Detection (Debug Mode)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Shuttle Detection (Debug Mode)", 1280, 720)
        cv2.imshow("YOLO Shuttle Detection (Debug Mode)", frame)

        # 按q退出，按空格暂停
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n用户退出")
            break
        elif key == ord(' '):
            print("\n暂停，按任意键继续")
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("\n调试结束")