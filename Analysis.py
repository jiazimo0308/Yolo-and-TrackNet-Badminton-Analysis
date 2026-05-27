#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/21
import os
import numpy as np
from scipy import stats
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False
from ChangDi import *
from Player import *
from SpeedAndDiss import *
from Map_chandi import *
from Pose import *
from Badminton import *


base_dir = "/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统"
folders = {
    "videos": f"{base_dir}/videos",
    "runs": f"{base_dir}/runs",
    "results": f"{base_dir}/results",
}
for path in folders.values():
    os.makedirs(path, exist_ok=True)

video_dir = folders["videos"]
runs_dir = folders["runs"]
results_dir = folders["results"]

video_path = '/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/videos/Video Project.mp4'
model_path='/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/yolov8n.pt'
pose_model_path='/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/yolov8n-pose.pt'
weights_path= '/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统/Track_Model/V3/model_best.pt'

#找场地边界并划定可判断区域
find_chang = FindChang(video_path,runs_dir)
court_coords = find_chang.select_4points(force_reslect=True)#force_reslect=True
mid_coords = find_chang.MiddleLine(force_reslect=True)#获取中线
#球员监测
find_player=PlayerDetector(court_coords,mid_coords,model_path)
find_player.reset_tracker()#对轨迹进行重置
#小地图映射器
mapper = Mapping_changdi(court_coords)
#羽毛球检测器
bad_detector = ShuttleDetector(weights_path,max_speed=1500)
#球员姿态监测
pose_detector = PoseDetector(court_coords, mid_coords, pose_model_path)
#球员速度监测
speed_diss=Speed(court_coords,30)


#################################################################
# 创建可缩放窗口
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#视频写入
output_save_path = os.path.join(folders["results"], "badminton_analysis_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_save_path, fourcc, fps, (frame_w, frame_h))
#地图的缩放
MINIMAP_W, MINIMAP_H = 180, 360
raw_bg = cv2.imread("./court_bg.png")
court_bg = cv2.resize(raw_bg, (MINIMAP_W, MINIMAP_H))
#
cv2.namedWindow("比赛预览", cv2.WINDOW_NORMAL)
cv2.resizeWindow("比赛预览", 1200, 700)

trajectory_history = {"P1": [], "P2": []}#轨迹缓存
latest_speed = {"P1": 0.0, "P2": 0.0}#速度缓存
count_diss = {"P1": 0.0, "P2": 0.0}#累计距离缓存
shuttle_trajectory = []#羽毛球轨迹
#中线坐标计算
p_mid1 = (int(mid_coords[0][0]), int(mid_coords[0][1]))
p_mid2 = (int(mid_coords[1][0]), int(mid_coords[1][1]))
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    clean_frame = frame.copy()

    players = find_player.track_frame(clean_frame, frame_idx)
    current_point,bad_trajectories= bad_detector.detect(clean_frame, fps)
    pose_keypoints = pose_detector.detect_pose_frame(clean_frame)
    speed_diss.set_trajectories(trajectory_history)
    speed_diss.calculate_speed()


    for label in ["P1", "P2"]:
        for s in reversed(speed_diss.speed_data.get(label, [])):
            if s["frame"] <= frame_idx:
                latest_speed[label] = s["speed_mps"]
                count_diss[label] = s["total_dist_m"]
                break

    ang1, str1 = pose_detector.pose_data["P1"]
    ang2, str2 = pose_detector.pose_data["P2"]

    #绘制界面
    cv2.polylines(frame, [court_coords.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
    cv2.line(frame, p_mid1, p_mid2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "NET-MIDDLE", (p_mid1[0] + 10, p_mid1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    #速度姿态等信息
    cv2.rectangle(frame, (10, 10), (380, 450), (0, 0, 0), -1)
    cv2.putText(frame, f"P1", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {latest_speed['P1']:.1f} m/s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"CDist: {count_diss['P1']:.1f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Body Angle: {ang1:.0f}*", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Leg Stretch Ratio:{str1:.1f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"P2", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    cv2.putText(frame, f"Speed: {latest_speed['P2']:.1f} m/s", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    cv2.putText(frame, f"CDist: {count_diss['P2']:.1f} m", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    cv2.putText(frame, f"Body Angle:{ang2:.0f}*", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    cv2.putText(frame, f"Leg Stretch Ratio:{str2:.1f}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    #球员轨迹
    for label in ["P1", "P2"]:
        color = (0, 255, 0) if label == "P1" else (0, 165, 255)
        for (f, x, y) in trajectory_history[label][-8:]:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    #球员框
    for box, foot_point, label in players:
        x1, y1, x2, y2 = box
        cx, cy = foot_point
        color = (0, 255, 0) if label == "P1" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if (cx, cy) not in trajectory_history[label]:
            trajectory_history[label].append((frame_idx, cx, cy))

    #小地图
    mini_map = court_bg.copy()
    MINIMAP_X = frame.shape[1] - MINIMAP_W - 20
    MINIMAP_Y = 20
    mapper.trajectories_in(trajectory_history)
    mapped_traj = mapper.track_mapping()
    for label in ["P1", "P2"]:
        if label not in mapped_traj: continue
        color = (0, 255, 0) if label == "P1" else (0, 165, 255)
        for (f, rx, ry) in mapped_traj[label]:
            px = int(5 + (rx / 7.1) * (MINIMAP_W - 10))
            py = int(5 + (ry / 15.0) * (MINIMAP_H - 10))
            cv2.circle(mini_map, (px, py), 2, color, -1)
        if len(mapped_traj[label]) > 0:
            f, rx, ry = mapped_traj[label][-1]
            px = int(5 + (rx / 7.1) * (MINIMAP_W - 10))
            py = int(5 + (ry / 15.0) * (MINIMAP_H - 10))
            cv2.circle(mini_map, (px, py), 6, color, -1)
            cv2.putText(mini_map, label, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    '''
        if current_point is not None:
        x_bad, y_bad = current_point
        rx_bad, ry_bad = mapper.map_badminton_with_height(x_bad, y_bad)
        px_bad = int(5 + (rx_bad / 7.1) * (MINIMAP_W - 10))
        py_bad = int(5 + (ry_bad / 15.0) * (MINIMAP_H - 10))
        cv2.circle(mini_map, (px_bad, py_bad), 4, (0, 0, 255), -1)
        cv2.putText(mini_map, "ball", (px_bad + 5, py_bad + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)'''

    frame[MINIMAP_Y:MINIMAP_Y + MINIMAP_H, MINIMAP_X:MINIMAP_X + MINIMAP_W] = mini_map

    #姿势骨架
    limbs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)]
    for label, kpts in pose_keypoints:
        color = (80, 255, 80) if label == "P1" else (0, 165, 225)
        for i, j in limbs:
            x1, y1 = kpts[i]
            x2, y2 = kpts[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        for x, y in kpts:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    #羽毛球和羽毛球轨迹
    if current_point is not None:
        shuttle_trajectory.append(current_point)
        # 保留最近5个点
        if len(shuttle_trajectory) > 5:
            shuttle_trajectory.pop(0)
        n = len(shuttle_trajectory)
        # 只画线，不画圆点！
        for i in range(n - 1):
            pt1 = shuttle_trajectory[i]
            pt2 = shuttle_trajectory[i + 1]
            # 线条渐变：后面细 → 前面粗
            thickness = int(1 + (i / n) * 5)
            cv2.line(frame, pt1, pt2, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.circle(frame, current_point, 4, (0, 0, 255), -1)

    out.write(frame)

    # 显示
    cv2.imshow("比赛预览", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


    frame_idx += 1

out.release()
cap.release()
cv2.destroyAllWindows()
find_player.save_trajectories(folders["results"])
speed_diss.save_speed_data(folders["results"])
print("\n运行完成！")
