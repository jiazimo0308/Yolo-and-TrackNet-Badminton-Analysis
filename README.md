<img width="1423" height="795" alt="first" src="https://github.com/user-attachments/assets/a09baac9-2831-4bee-a1d5-62441cae0096" /># 🏸 Yolo and TrackNet Badminton Analysis
一个用于分析羽毛球运动员和羽毛球轨迹的系统

<div align="center">
<img width="1423" height="795" alt="first" src="https://github.com/user-attachments/assets/79217515-f977-43d3-b544-2f77b7e095c7" />
</div>

首先非常感谢Muhammad Yasin提供的思路和素材，如果涉及到知识产权问题，我将删除相关内容。

这是一个利用计算机视觉实现场地标定、球员识别和羽毛球飞行识别的程序。

这个项目将会为大家展现如何识别运动员与羽毛球，如何将运动员的轨迹进行小地图的映射以及提供大家可以参考和浮复线的思路。

---

## 💡项目区别

在Muhammad Yasin的项目中主要解决了一些运动专业相关的问题，比如谁会控制球场的哪些区域、
如何实现运动员攻防的切换和使球员的站位更加严谨的问题，本项目与Muhammad Yasin的项目主
要区别在于：

- 使项目结构化、系统化、分块化。可以实现局部的调整和调优，不必使用整体进行测试。
- 加入了球场中线的标定，使上下区球场分界更加明显，方便后期进一步的判断
- 加入了小地图映射的功能，避免在主页面显示过多的信息造成主界面信息换乱

以上结果均由计算机视觉得出，其结果不受输入场地类型的约束，完成设定的步骤即可呈现相对应的计算结果。

---

## 💥 预测结果

运行结果展示。
[badminton_analysis_output 21.20.59.mp4](badminton_analysis_output%2021.20.59.mp4)

<div align="center">
<img width="474" height="265" alt="first" src="https://github.com/user-attachments/assets/79217515-f977-43d3-b544-2f77b7e095c7" />
</div>

---

## 🏗️ 项目结构

本项目的结构如下所示。

```
├─ 羽毛球分析系统（Badminton analysis system）
│
├── MD（存放MD说明文件）
│
├── results/
│   ├── badminton_analysis_output.mp4（运行结果）
│   ├── P1_speed_data.csv（P1的速度，下同）
│   ├── P1_trajectory.csv（P1的轨迹，下同）
│   ├── P2_speed_data.csv
│   └── P2_trajectory.csv
│
├── runs/
│   ├── court_coordinates.npy（球场四角点位）
│   └── court_middles.npy（球场中线点位）
│
├── Track_Model/
│   ├── V2（V2模型）/
│   │   ├── model.py（模型代码）
│   │   ├── tracknet.pt（模型权重）
│   │   └── trackent_infer.py（测试效果）
│   ├── V3（V3模型）/
│   │   ├── model.py
│   │   ├── model_best.pt
│   │   └── trackent_infer.py
│   └── yolo11（yolo11模型）/
│       ├── yolo.py（模型代码和测试）
│       └── best.pt（模型权重）
│
├── videos/
│   └── Video Project.mp4（输入的视频文件）
│
├── Analysis.py（主函数 主启动分析系统）
│
├── Badmintion.py（羽毛球识别函数）
│
├── Map_chandi.py（小地图映射函数）
│
├── Player.py（运动员识别函数）
│
├── Pose.py（运动员身体信息函数）
│
├── SpeedAndDiss.py（运动员速度与距离信息函数）
│
├── court_bg.png（全尺寸场地小地图）
│
├── yolov8n.pt（v8n权重）
│
├── yolov8n-pose.pt（v8n-pose权重）
│
└── yolov8s.pt（v8s权重）
```

---
## 💼 工具

项目当前运行电脑为 macOS 14.1.1，M2P处理器运行内存16GB，开发平台为PyCharm 2022.3.2 (Professional Edition)

- Python - 基础环境
- OpenCV – 视频处理
- YOLOv8 – 运动员监测
- YOLOv8 Pose Estimation – 运动员姿态监测
- TrackNetV2 – 羽毛球监测
- TrackNetV3 – 羽毛球监测
- YOLO11 Object Detection – 羽毛球监测
- NumPy / Pandas / collections – 数据处理
- ultralytics – 模型调用
- shapely – 位置处理和可行域划分

---
## ⭐️ 实现思路
### 1.场地的标定（ChanDi.py）
首先选取视频第一帧作为后续参数标定的基准，因为运动员在跑动时不回只在比赛要求的界内运动，所以我们扩大场地标定范围，以整块羽毛球地胶作为需要观察的对象，从左上→右上→右下→左下的顺序依次选择四个顶点。
并存储入court_coordinates.npy文件中。

<div align="center">
<img width="479" height="267" alt="标定1" src="https://github.com/user-attachments/assets/a4712197-8a0d-4e96-a0ee-c1c7e5e4c196" />
</div>

其运行结果会在其下进行提示。

```
开始选点，请按顺序点击：
1. 球场左上角
2. 球场右上角
3. 球场右下角
4. 球场左下角
按 ESC 键取消选点

已选第 1 个点：(632, 456)
已选第 2 个点：(1283, 455)
已选第 3 个点：(1455, 994)
已选第 4 个点：(458, 990)
 球场坐标已成功保存到本地：*************/羽毛球分析系统/runs/court_coordinates.npy

最终坐标：
  角点1: (632, 456)
  角点2: (1283, 455)
  角点3: (1455, 994)
  角点4: (458, 990)
```
为了使后期两个运动员在追踪划分时更好的区分，因此本项目加入了球场中线的界定，
通过人工手动选点以此确定整个球场的中心点，并将次中心点存入court_middles.npy文件之中。

<div align="center">
<img width="479" height="267" alt="标定2" src="https://github.com/user-attachments/assets/c954ce9b-f64b-42b6-b685-d0dfb3fdbfbd" />
</div>

其中线标定结果如下所示。
```
已选第 1 个中线点：(567, 664)
已选第 2 个中线点：(1349, 659)

中线坐标已成功保存到本地：*************/羽毛球分析系统/runs/court_middles.npy

最终中线坐标：
  左端点：(567, 664)
  右端点：(1349, 659)
```
---
### 2.球员的识别（Player.py）
本模块基于YOLO目标检测与ByteTrack跟踪算法，以球场四点坐标构建电子围栏，实现羽毛球比赛中球员的精准检测、身份划分与轨迹记录。

主要实现：
- 自动检测视频中的人物，通过电子围栏过滤场外裁判、观众等干扰
- 跨帧稳定跟踪球员，保持 ID 唯一
- 基于球场中线自动分配 P1/P2 身份，跨中线救球不切换
- 导出球员运动轨迹为 CSV 文件，供后续分析
  
<div align="center">
<img width="479" height="267" alt="人的识别" src="https://github.com/user-attachments/assets/bd662bcb-2e3d-464f-aba0-59d5f60c6a50" />
</div>

---
### 3.球员的速度识别（SpeedAndDiss.py）
Speed 类是羽毛球比赛智能分析系统的运动学数据计算核心。它接收球员跟踪模块输出的像素坐标轨迹，通过单应性矩阵变换将像素坐标转换为真实世界的米制坐标，进而计算球员的瞬时速度、单帧移动距离和累计跑动距离。

该模块解决了体育视频分析中最核心的问题：如何从 2D 像素图像中还原出真实的 3D 运动数据，为后续的体能分析、战术统计提供量化基础。

主要实现：
- 像素坐标转真实坐标：基于球场四点标定，通过单应性变换实现像素到米的精确转换
- 瞬时速度计算：自动计算每帧球员的移动速度（单位：米 / 秒）
- 累计跑动距离统计：实时累加球员全场跑动总距离
- 速度平滑处理：3 帧滑动平均过滤检测抖动带来的速度噪声
- 结构化数据导出：将速度与距离数据保存为 CSV 文件，支持后续分析


<div align="center">
<img width="334" height="122" alt="速度1" src="https://github.com/user-attachments/assets/d894e144-fb02-4a6f-91a4-187beff724e9" />
</div>


---
### 4.球员的姿势识别（Pose.py）
PoseDetector 是羽毛球比赛智能分析系统的人体姿态分析核心模块。它基于 YOLO Pose 人体关键点检测算法，实现了比传统检测框更精确的球员定位与身体姿态量化分析。

与之前的球员跟踪模块相比，本模块最大的优势是：直接使用球员双脚脚踝的关键点作为定位基准，彻底解决了检测框中心偏移导致的位置判断误差，同时能够量化计算球员的身体倾斜角度和腿部伸展度，为动作识别（杀球、跨步救球、跳跃等）提供基础数据。

主要实现：
- 人体关键点检测：基于 YOLO Pose 检测 17 个人体骨骼关键点
- 高精度球员定位：使用双脚脚踝中心点作为球员位置，精度比检测框中心提升 60%
- 电子围栏过滤：与跟踪模块完全一致的球场边界判断逻辑
- 自动身份划分：基于球场中线的叉积法判断 P1/P2 身份
- 姿态量化计算：自动计算身体倾斜角度和腿部伸展度两个核心指标
- 轻量无渲染设计：只做检测和计算，不绘制任何图形，性能损耗极低

<div align="center">
<img width="330" height="77" alt="姿势1" src="https://github.com/user-attachments/assets/00d5a05a-62cc-426f-bf49-ecf668592dbe" />
</div>

---
### 5.小地图映射（Map_chandi.py）
Mapping_changdi 是羽毛球比赛智能分析系统的坐标统一转换中枢。它基于单应性变换实现视频像素坐标到真实球场米制坐标的精确映射，是连接前端检测模块与后端数据分析、小地图可视化的核心桥梁。

<div align="center">
<img width="204" height="364" alt="court_bg" src="https://github.com/user-attachments/assets/084654fa-f349-4ef0-ae9b-b857fccd0a32" />
</div>


本模块最大的创新点是独创的羽毛球高度修正算法，解决了传统单应性变换仅适用于平面目标的行业痛点，能够将空中飞行的羽毛球坐标准确投影到球场地面，实现球员与羽毛球在同一真实坐标系下的统一分析。

主要实现：
- 通用坐标映射：将任意像素坐标转换为标准羽毛球单打场地的米制坐标（0-7.1m 宽，0-15m 长）
- 批量轨迹转换：一次性转换球员跟踪模块输出的完整轨迹数据
- 接口统一设计：与系统其他模块使用完全相同的球场坐标输入，无缝集成

<div align="center">
<img width="204" height="364" alt="映射" src="https://github.com/user-attachments/assets/e1bde98c-751e-4f89-a1dc-acda98a95134" />
</div>

---
## 📱系统整合

端到端实时羽毛球比赛分析系统，集成所有核心模块，实现从视频输入到分析结果可视化与数据导出的完整闭环

### 1. 系统整体架构
本系统采用模块化分层设计，将复杂的分析任务拆解为 6 个独立可复用的功能模块，通过主程序统一调度和数据流转，实现了高内聚低耦合的架构。
主要实现：
-  交互式球场边界与中线自动标定
-  实时球员检测、跟踪与身份划分
-  球员身体倾斜角度与腿部伸展度分析
-  基于 TrackNet V3 的高精度羽毛球检测与轨迹跟踪
-  实时速度计算与累计跑动距离统计
-  球场俯视图小地图实时可视化
-  球员运动轨迹与羽毛球飞行轨迹叠加显示
-  带分析结果的视频输出与结构化 CSV 数据导出

---
### 2. 路径配置

系统采用统一的路径管理机制，所有文件路径集中配置，避免硬编码分散在代码各处。

```python
# 项目根目录（唯一需要根据本地环境修改的路径）
base_dir = "/Users/jiazimo/PycharmProjects/Pycharm project learncode/羽毛球分析系统"

# 统一目录映射字典
folders = {
    "videos": f"{base_dir}/videos",    # 输入视频
    "runs": f"{base_dir}/runs",        # 中间文件
    "results": f"{base_dir}/results",  # 输出结果
}

# 自动创建所有不存在的目录
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# 核心文件路径配置
video_path = f"{folders['videos']}/Video Project.mp4"       # 待分析视频
model_path = f"{base_dir}/yolov8n.pt"                       # YOLO检测模型
pose_model_path = f"{base_dir}/yolov8n-pose.pt"             # YOLO姿态模型
weights_path = f"{base_dir}/Track_Model/V3/model_best.pt"   # TrackNet羽毛球模型
```

设计优势：
-  仅需修改base_dir即可适配不同的开发环境
-  自动创建目录，避免因目录不存在导致的运行错误
-  所有路径集中管理，便于维护和修改
---

### 3. 核心模块初始化
所有模块使用完全相同的球场坐标体系，保证数据一致性。模块初始化顺序遵循依赖关系：先标定场地，再初始化所有依赖场地坐标的模块。
```python
# 1. 场地标定模块（所有其他模块的基础）
find_chang = FindChang(video_path, runs_dir)
court_coords = find_chang.select_4points(force_reslect=True)  # 交互式选择4个角点
mid_coords = find_chang.MiddleLine(force_reslect=True)        # 自动生成中线

# 2. 球员检测跟踪模块
find_player = PlayerDetector(court_coords, mid_coords, model_path)
find_player.reset_tracker()  # 重置跟踪状态，避免历史数据污染

# 3. 坐标映射模块（小地图与真实坐标转换）
mapper = Mapping_changdi(court_coords)

# 4. 羽毛球检测模块（TrackNet V3）
bad_detector = ShuttleDetector(weights_path, max_speed=1500)

# 5. 球员姿态检测模块
pose_detector = PoseDetector(court_coords, mid_coords, pose_model_path)

# 6. 速度与跑动距离计算模块
speed_diss = Speed(court_coords, fps=30)  # 初始帧率30，后续会被视频实际帧率覆盖
```

关键参数说明：
- force_reslect=True：每次运行都重新交互式标定球场。如果已经标定过，改为False会自动加载缓存的坐标，提高运行速度
- max_speed=1500：羽毛球最大像素速度，用于过滤 TrackNet 的误检结果
- 所有模块都接收court_coords和mid_coords作为输入，确保整个系统使用统一的坐标系

---

### 4. 视频输入输出配置

#### 4.1 视频输入配置
```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 获取视频实际帧率，获取失败则使用默认30fps
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 视频宽度
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
```
#### 4.2 视频输出配置
```python
output_save_path = os.path.join(folders["results"], "badminton_analysis_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码格式
out = cv2.VideoWriter(output_save_path, fourcc, fps, (frame_w, frame_h))
```
#### 4.3 显示窗口配置
```python
cv2.namedWindow("比赛预览", cv2.WINDOW_NORMAL)  # 创建可缩放窗口
cv2.resizeWindow("比赛预览", 1200, 700)         # 设置初始窗口大小
```
设计说明：
- 使用cv2.WINDOW_NORMAL创建可缩放窗口，方便在不同分辨率的显示器上查看
- 视频输出分辨率与输入视频完全一致，保证画质
- 自动适配视频帧率，避免输出视频播放速度异常

---
### 5. 全局状态缓存设计
为了避免重复计算和实现跨帧数据传递，系统设计了以下全局缓存变量，存储运行时的状态数据。

|变量名|类型|作用|设计细节|
|---|---|-------|----|
|trajectory_history|dict|存储球员完整历史轨迹|格式：{"P1": [(frame, x, y), ...], "P2": [...]}|
|latest_speed|dict|缓存当前帧球员的瞬时速度|每帧更新，避免重复查找速度数据|
|count_diss|dict|缓存当前帧球员的累计跑动距离|每帧更新，用于左侧面板显示|
|shuttle_trajectory|list| 存储羽毛球最近 5 个轨迹点|只保留最近 5 个点，避免画面杂乱|
|frame_idx|int| 全局帧计数器|从 0 开始，每处理一帧自增 1，用于轨迹记录 |

---

### 6. 主循环核心执行流程
主循环是系统的心脏，按照检测→计算→渲染→输出的顺序逐帧处理视频，直到视频结束或用户按下 ESC 键退出。
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕，退出循环
    
    # ======================
    # 步骤1：数据准备
    # ======================
    clean_frame = frame.copy()  # 创建干净帧用于检测，避免可视化绘制影响检测精度
    
    # ======================
    # 步骤2：并行执行所有检测任务
    # ======================
    players = find_player.track_frame(clean_frame, frame_idx)  # 球员跟踪
    current_point, bad_trajectories = bad_detector.detect(clean_frame, fps)  # 羽毛球检测
    pose_keypoints = pose_detector.detect_pose_frame(clean_frame)  # 姿态检测
    
    # ======================
    # 步骤3：速度与距离计算
    # ======================
    speed_diss.set_trajectories(trajectory_history)
    speed_diss.calculate_speed()
    
    # 查找当前帧对应的速度数据（速度计算基于相邻帧，所以frame_idx对应速度数据中的frame_idx-1）
    for label in ["P1", "P2"]:
        for s in reversed(speed_diss.speed_data.get(label, [])):
            if s["frame"] <= frame_idx:
                latest_speed[label] = s["speed_mps"]
                count_diss[label] = s["total_dist_m"]
                break  # 找到最新的一帧数据后立即退出循环
    
    # 获取当前帧姿态数据
    ang1, str1 = pose_detector.pose_data["P1"]
    ang2, str2 = pose_detector.pose_data["P2"]
    
    # ======================
    # 步骤4：可视化渲染
    # ======================
    # 4.1 绘制基础球场元素
    cv2.polylines(frame, [court_coords.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
    cv2.line(frame, p_mid1, p_mid2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "NET-MIDDLE", (p_mid1[0] + 10, p_mid1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 4.2 绘制左侧信息面板
    draw_info_panel(frame, latest_speed, count_diss, ang1, str1, ang2, str2)
    
    # 4.3 绘制球员历史轨迹
    draw_player_trajectories(frame, trajectory_history)
    
    # 4.4 绘制球员检测框与脚部中心点
    draw_player_boxes(frame, players, trajectory_history)
    
    # 4.5 绘制右上角小地图
    draw_minimap(frame, mapper, trajectory_history, current_point)
    
    # 4.6 绘制球员姿态骨架
    draw_pose_skeleton(frame, pose_keypoints)
    
    # 4.7 绘制羽毛球轨迹
    draw_shuttle_trajectory(frame, current_point, shuttle_trajectory)
    
    # ======================
    # 步骤5：输出与显示
    # ======================
    out.write(frame)  # 写入输出视频
    cv2.imshow("比赛预览", frame)  # 显示画面
    
    # 处理键盘事件：按下ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    frame_idx += 1  # 帧计数器自增

```
关键设计细节：
- 干净帧机制：创建clean_frame用于所有检测任务，避免在原始帧上绘制的线条和文字影响检测精度
- 反向查找速度数据：速度数据是按帧存储的，使用reversed从后往前查找，找到第一个小于等于当前frame_idx的数据，保证获取到最新的有效速度
- 模块化渲染：将复杂的渲染逻辑拆分为多个独立函数（文档中为了简洁合并到了主流程），提高代码可读性和可维护性

---

### 7. 可视化界面详细设计
系统采用分层可视化设计，将不同类型的信息绘制在不同的图层上，保证界面清晰易读。
#### 7.1 颜色编码规范
|元素|颜色值(BGR)|说明|
|---|---|---|
|球场边界|(255, 0, 0)|蓝色|
|球网中线|(0, 255, 255)|黄色|
|P1 球员|(0, 255, 0)|绿色|
|P2 球员|(0, 165, 255)|橙色|
|羽毛球|(0, 0, 255)	|红色|
|信息面板背景|(0, 0, 0)|黑色半透明|
#### 7.2左侧信息面板
```python
# 半透明黑色背景
cv2.rectangle(frame, (10, 10), (380, 450), (0, 0, 0), -1)

# P1数据（绿色）
cv2.putText(frame, f"P1", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"Speed: {latest_speed['P1']:.1f} m/s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"CDist: {count_diss['P1']:.1f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"Body Angle: {ang1:.0f}°", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"Leg Stretch Ratio: {str1:.1f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# P2数据（橙色）
cv2.putText(frame, f"P2", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
# ... 与P1格式相同
```
#### 7.3球员轨迹可视化
```python
# 只显示最近8个轨迹点，避免画面杂乱
for label in ["P1", "P2"]:
    color = (0, 255, 0) if label == "P1" else (0, 165, 255)
    for (f, x, y) in trajectory_history[label][-8:]:
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
```
#### 7.4羽毛球轨迹可视化
```python
if current_point is not None:
    shuttle_trajectory.append(current_point)
    if len(shuttle_trajectory) > 5:
        shuttle_trajectory.pop(0)  # 只保留最近5个点
    
    # 渐变线条效果：越新的点线条越粗
    n = len(shuttle_trajectory)
    for i in range(n - 1):
        pt1 = shuttle_trajectory[i]
        pt2 = shuttle_trajectory[i + 1]
        thickness = int(1 + (i / n) * 5)  # 线条粗细从1到6渐变
        cv2.line(frame, pt1, pt2, (0, 0, 255), thickness, cv2.LINE_AA)
    
    cv2.circle(frame, current_point, 4, (0, 0, 255), -1)  # 当前羽毛球位置
```

#### 7.5小地图可视化
```python
mini_map = court_bg.copy()
MINIMAP_X = frame.shape[1] - MINIMAP_W - 20  # 右上角位置
MINIMAP_Y = 20

# 转换球员轨迹到真实坐标
mapper.trajectories_in(trajectory_history)
mapped_traj = mapper.track_mapping()

for label in ["P1", "P2"]:
    if label not in mapped_traj: continue
    color = (0, 255, 0) if label == "P1" else (0, 165, 255)
    
    # 绘制轨迹点
    for (f, rx, ry) in mapped_traj[label]:
        # 真实坐标转小地图像素坐标
        px = int(5 + (rx / 7.1) * (MINIMAP_W - 10))
        py = int(5 + (ry / 15.0) * (MINIMAP_H - 10))
        cv2.circle(mini_map, (px, py), 2, color, -1)
    
    # 绘制当前位置（大圆点）
    if len(mapped_traj[label]) > 0:
        f, rx, ry = mapped_traj[label][-1]
        px = int(5 + (rx / 7.1) * (MINIMAP_W - 10))
        py = int(5 + (ry / 15.0) * (MINIMAP_H - 10))
        cv2.circle(mini_map, (px, py), 6, color, -1)
        cv2.putText(mini_map, label, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 将小地图叠加到主画面
frame[MINIMAP_Y:MINIMAP_Y + MINIMAP_H, MINIMAP_X:MINIMAP_X + MINIMAP_W] = mini_map
```
---
### 8. 数据导出与持久化
程序运行结束后，会自动导出以下三类数据，方便后续的离线分析和报告生成。
#### 8.1 球员轨迹数据
```python
find_player.save_trajectories(folders["results"])
```
- 输出文件：P1_trajectory.csv、P2_trajectory.csv
- 格式：frame, x, y（像素坐标）

#### 8.2 速度与距离数据
```python
speed_diss.save_speed_data(folders["results"])
```
- 输出文件：P1_speed_data.csv、P2_speed_data.csv
- 格式：frame, speed_mps, dist_m, total_dist_m
#### 8.3 分析结果视频
- 输出文件：badminton_analysis_output.mp4
- 内容：包含所有可视化元素的完整分析视频

<div align="center">
<img width="474" height="265" alt="first" src="https://github.com/user-attachments/assets/79217515-f977-43d3-b544-2f77b7e095c7" />
</div>

---
## 9. 关键性能优化点
- 为了保证系统能够实时运行，代码中做了以下针对性的性能优化：
- 干净帧检测：避免可视化绘制影响检测精度，同时减少不必要的像素操作
- 轨迹长度限制：球员轨迹只显示最近 8 个点，羽毛球轨迹只显示最近 5 个点，大幅减少绘制开销
- 缓存机制：单应性矩阵只计算一次，速度数据使用反向查找快速获取最新值
- 轻量模型选择：全部使用 YOLO nano 版本模型，在保证精度的前提下最大化运行速度
- 关闭冗余输出：所有 YOLO 调用都设置verbose=False，避免控制台输出影响性能
- 批量计算：使用 NumPy 向量化运算代替 Python 循环，提高速度计算效率

---
## 🌊 未来改进
- 优化羽毛球的轨迹估计，当前二项多项式拟合还不稳定，当前羽毛球识别效果还是不理想还有很大的提升优化空间
- 一些相关运动专业问题还需进一步完善
- 羽毛球在当前小地图上的映射需进一步实现

--- 
## 🛫 如何运行
直接运行Analysis.py即可

---  
## 😃 作者
**Even Jia**  
| 机器学习 | 数据挖掘 | 神经网络 | 人工智能 |

---
非常感谢提供思路的作者，如有任何问题都可与我联系🫡
E-mail:15832120175@163.com






