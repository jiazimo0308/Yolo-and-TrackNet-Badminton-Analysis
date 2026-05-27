#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/25

import torch
import cv2
import numpy as np
from model import TrackNetV2 as TrackNet

MODEL_PATH = "/Track_Model/V3/model_best.pt"
VIDEO_PATH = "*******************************/羽毛球分析系统/videos/Video Project.mp4"

def get_shuttle_position(bin_img):
    """获取羽毛球中心点（V3 专用）"""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, 0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 2 or area > 50:
        return 0, 0, 0
    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0, 0, 0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return 1, int(cx), int(cy)

pos_history = []
SMOOTH_FRAMES = 1
def smooth_pos(x, y):
    global pos_history
    if x > 0 and y > 0:
        pos_history.append((x, y))
    if len(pos_history) > SMOOTH_FRAMES:
        pos_history.pop(0)
    if not pos_history:
        return 0, 0
    xs = [p[0] for p in pos_history]
    ys = [p[1] for p in pos_history]
    return int(np.mean(xs)), int(np.mean(ys))

def main():
    imgsz = (288, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("使用设备:", device)

    model = TrackNet(in_dim=9, out_dim=3)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    q_imgs = []
    count = 0
    last_x, last_y = 0, 0

    print("▶️  开始 TrackNetV3 预测，按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1

        # 预处理（直接使用完整原图）
        img = cv2.resize(frame, (imgsz[1], imgsz[0]))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0

        q_imgs.append(img)
        if len(q_imgs) < 3:
            continue
        if len(q_imgs) > 3:
            q_imgs.pop(0)

        # V3 输入：3帧堆叠 = 9 通道
        imgs_torch = torch.cat(q_imgs, dim=0).unsqueeze(0).to(device)

        # V3 推理
        with torch.no_grad():
            preds = model(imgs_torch)
            preds = preds.cpu().numpy()

        # V3 输出：取第 1 张热力图（中间帧预测）
        pred_heatmap = preds[0][1]
        thresh = np.max(pred_heatmap) * 0.45
        pred_bin = (pred_heatmap > thresh).astype(np.uint8) * 255
        visible, cx_pred, cy_pred = get_shuttle_position(pred_bin)

        # 坐标映射（完整图映射，无裁剪）
        if visible:
            cx = int(round(cx_pred * original_w / imgsz[1]))
            cy = int(round(cy_pred * original_h / imgsz[0]))
            last_x, last_y = cx, cy
        else:
            cx, cy = last_x, last_y

        # 轨迹平滑
        smooth_x, smooth_y = smooth_pos(cx, cy)

        # 显示
        show_frame = frame.copy()
        if smooth_x > 0 and smooth_y > 0:
            cv2.circle(show_frame, (smooth_x, smooth_y), 8, (0, 0, 255), -1)
            cv2.circle(show_frame, (smooth_x, smooth_y), 10, (255, 255, 255), 1)

        # 显示预测结果
        cv2.imshow("TrackNetV3 预测结果", show_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count % 50 == 0:
            print(f"已处理 {count} 帧")

    cap.release()
    cv2.destroyAllWindows()
    print("预测结束")

if __name__ == "__main__":
    main()
