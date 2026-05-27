#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jiazimo
# @Time : 2026/5/21
import torch
import cv2
import numpy as np
from Track_Model.V2.model import TrackNet

# ====================== 路径 ======================
MODEL_PATH = "/Track_Model/V2/tracknet.pt"
VIDEO_PATH = "****************************/羽毛球分析系统/videos/Video Project.mp4"
COORD_NPY = "****************************/羽毛球分析系统/runs/court_coordinates.npy"
# ===================================================

# ===================== 自动取正确左右边界 =====================
court = np.load(COORD_NPY).astype(np.int32)
x3 = court[2][0]
x4 = court[3][0]

# 自动保证 left < right，永远不报错
crop_left = min(x3, x4)
crop_right = max(x3, x4)
print(f"✅ 裁剪左右边界：left={crop_left}, right={crop_right}")
# ==============================================================

def get_shuttle_position(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (0, 0, 0)
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return (0, 0, 0)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (1, int(cx), int(cy))

def main():
    imgsz = (288, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("使用设备:", device)

    model = TrackNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    q_imgs = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1

        # ===================== 只裁左右，高度不动 =====================
        cropped_frame = frame[:, crop_left:crop_right]
        # ============================================================

        # 预处理
        img = cv2.resize(cropped_frame, (imgsz[1], imgsz[0]))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0

        q_imgs.append(img)
        if len(q_imgs) < 3:
            continue
        if len(q_imgs) > 3:
            q_imgs.pop(0)

        imgs_torch = torch.cat(q_imgs, dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(imgs_torch)
            preds = preds.cpu().numpy()

        pred_heatmap = preds[0][1]
        pred_bin = (pred_heatmap > 0.5).astype(np.uint8) * 255
        visible, cx_pred, cy_pred = get_shuttle_position(pred_bin)

        # 坐标映射回原图
        crop_w = crop_right - crop_left
        if visible:
            cx = crop_left + int(cx_pred * crop_w / imgsz[1])
            cy = int(cy_pred * original_h / imgsz[0])
        else:
            cx, cy = 0, 0

        # 显示
        show_frame = frame.copy()
        if visible:
            cv2.circle(show_frame, (cx, cy), 8, (0, 0, 255), -1)

        cv2.imshow("Cropped Frame", cropped_frame)
        cv2.imshow("Result", show_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 结束")

if __name__ == "__main__":
    main()
