#!/usr/bin/env python
# -*- coding: utf-8 -*-            
# @Author : Jiazimo
# @Time : 2026/5/21
import os
import cv2
import numpy as np

class FindChang():
    def __init__(self, video_path,results_dir):
        self.video_path = video_path
        self.save_path1 = f"{results_dir}/court_coordinates.npy"
        self.save_path2 = f"{results_dir}/court_middles.npy"
        self.src_pts = []
        self.mid_pts = None  # 中线2个端点
        self.img_copy = None

    def get_first_frame(self):
        '''选取视频第一帧'''
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("无法读取视频第一帧！")
        return frame

    def on_mouse(self, event, x, y, flags, param):
        """鼠标点击事件：选4个场地角点 + 实时画红点+序号"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.src_pts) < 4:
            self.src_pts.append([x, y])
            print(f"已选第 {len(self.src_pts)} 个点：({x}, {y})")
            # 实时画出红色圆点和序号
            cv2.circle(self.img_copy, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(self.img_copy, str(len(self.src_pts)), (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def select_4points(self, force_reslect=False):
        """
        显示第一帧，鼠标选点（顺序：左上→右上→右下→左下）
        :param force_reslect: 强制重新选点，即使本地已有保存的坐标
        :return: 4个角点坐标 (4,2) float32数组
        """
        # 第一步：先检查本地是否有保存的坐标
        if os.path.exists(self.save_path1) and not force_reslect:
            try:
                self.src_pts = np.load(self.save_path1).tolist()
                print(f"自动加载本地保存的球场坐标：{self.save_path1}")
                print("当前坐标：")
                for i, (x, y) in enumerate(self.src_pts):
                    print(f"  角点{i+1}: ({x}, {y})")
                print("如需重新选点，请调用 select_points(force_reslect=True)")
                return np.float32(self.src_pts)
            except Exception as e:
                print(f"加载本地坐标失败：{e}，将重新进行选点")
        frame = self.get_first_frame()
        self.img_copy = frame.copy()
        self.src_pts = []  # 清空之前的点
        window_name = "球场标定 - 依次点击：左上→右上→右下→左下"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 700)  # 固定窗口大小方便选点
        cv2.setMouseCallback(window_name, self.on_mouse)
        print("\n开始选点，请按顺序点击：")
        print("1. 球场左上角")
        print("2. 球场右上角")
        print("3. 球场右下角")
        print("4. 球场左下角")
        print("按 ESC 键取消选点\n")
        # 选够4个点才退出
        while len(self.src_pts) < 4:
            cv2.imshow(window_name, self.img_copy)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                cv2.destroyAllWindows()
                raise Exception("用户取消了选点操作")
        cv2.destroyAllWindows()
        # 第三步：选点完成后自动保存到本地
        np.save(self.save_path1, np.array(self.src_pts, dtype=np.int32))
        print(f"\n 球场坐标已成功保存到本地：{self.save_path1}")
        print("最终坐标：")
        for i, (x, y) in enumerate(self.src_pts):
            print(f"  角点{i + 1}: ({x}, {y})")
        return np.float32(self.src_pts)

    def MiddleLine(self, force_reslect=False):
        """
        手动标定球场中线（球网）的两个端点
        :param force_reslect: 强制重新选点，即使本地已有保存的坐标
        :return: 中线两个端点坐标 (2,2) float32数组
        """
        # 第一步：检查本地是否有保存的中线坐标
        if os.path.exists(self.save_path2) and not force_reslect:
            try:
                self.mid_pts = np.load(self.save_path2).tolist()
                print(f"自动加载本地保存的中线坐标：{self.save_path2}")
                print(f"中线左端点：({self.mid_pts[0][0]}, {self.mid_pts[0][1]})")
                print(f"中线右端点：({self.mid_pts[1][0]}, {self.mid_pts[1][1]})")
                print("如需重新选点，请调用 MiddleLine(force_reslect=True)\n")
                return np.float32(self.mid_pts)
            except Exception as e:
                print(f"加载本地中线坐标失败：{e}，将重新进行选点")
        frame = self.get_first_frame()
        img_copy = frame.copy()
        mid_pts_temp = []
        window_name = "中线标定 - 依次点击：球网左端点→球网右端点"
        def on_mouse_mid(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(mid_pts_temp) < 2:
                mid_pts_temp.append([x, y])
                print(f"已选第 {len(mid_pts_temp)} 个中线点：({x}, {y})")
                cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(img_copy, str(len(mid_pts_temp)), (x + 15, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 700)
        cv2.setMouseCallback(window_name, on_mouse_mid)

        while len(mid_pts_temp) < 2:
            cv2.imshow(window_name, img_copy)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                raise Exception("用户取消了中线标定操作")

        cv2.destroyAllWindows()
        self.mid_pts = mid_pts_temp
        np.save(self.save_path2, np.array(self.mid_pts, dtype=np.int32))
        print(f"\n中线坐标已成功保存到本地：{self.save_path2}")
        print("最终中线坐标：")
        print(f"  左端点：({self.mid_pts[0][0]}, {self.mid_pts[0][1]})")
        print(f"  右端点：({self.mid_pts[1][0]}, {self.mid_pts[1][1]})")

        return np.float32(self.mid_pts)

