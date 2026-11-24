# IESKF-Voxel-LIO

基于 ROS 2 的激光惯性里程计。项目使用 Eigen `float32` 实现迭代误差状态卡尔曼滤波（IESKF），结合 IMU 传播、逐点去畸变和自适应体素平面地图，输出位姿、轨迹与可视化结果。

算法核心不依赖 IKFoM，也不使用 FAST-LIO1 的扫描线特征提取。ROS 2 输入层与滤波、IMU 处理和地图模块分离，关键时序与数学代码附有中文注释。

## 可视化

点云与轨迹：

![点云与轨迹可视化](docs/images/example1.gif)

体素平面：

![体素平面可视化](docs/images/example2.gif)

演示中可通过 RViz 的 `World scan` → `Decay Time` 暂时累积显示点云。`cloud_world` 话题本身是当前帧点云；项目当前没有发布完整的历史全局点云地图。

## 算法流程

1. 接收 LiDAR 与 IMU 数据，按时间同步；使用静止 IMU 数据初始化重力方向和陀螺仪零偏。
2. IMU 前向传播状态和协方差，记录扫描期间的运动轨迹。
3. 利用逐点时间将点云反向去畸变到扫描帧末。
4. 降采样后，将点与历史体素地图中的平面关联，构建点到平面观测并迭代更新状态。
5. 更新成功后，将点插入自适应体素地图，发布里程计、点云、轨迹、平面和诊断。

误差状态为 `[p, θ, θ_IL, t_IL, v, b_g, b_a, g_dir]`，共 23 维；固定外参时求解 17 维活动状态。状态与地图使用 `float32`，时间使用整数纳秒。

静止初始化可对齐横滚和俯仰；航向不能由重力单独确定，初始航向采用约定值。数据开头若处于运动状态，算法会等待后续静止窗口，之前的数据不产生轨迹。

## 环境依赖

- Ubuntu / ROS 2 Humble、C++17、CMake、colcon
- Eigen3、PCL、GoogleTest
- `livox_ros_driver2` 消息包；当前 ROS 构建即使使用其他雷达也需要该依赖
- RViz 2 和 `rviz_default_plugins`，用于可视化

下文命令以本机工作区 `/home/ros/ros_ws/fast-lio2-ws` 为例。本机 `livox_ros_driver2` 已安装在 `/home/ros/ros_ws/livox_ros/install`；其他机器请将路径替换为自己的安装路径。

## 编译与测试

在 ROS 2 工作区根目录执行：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash

colcon build --packages-select ieskf_voxel_lio --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

如果编译时内存不足，可在 `colcon build` 前设置 `CMAKE_BUILD_PARALLEL_LEVEL=1`。

运行单元测试：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
ctest --test-dir build/ieskf_voxel_lio --output-on-failure
```

仅编译核心库时不需要 ROS 或 PCL：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
cmake -S src/IESKF-Voxel-LIO -B build/ieskf_core -DLIO_BUILD_ROS=OFF
cmake --build build/ieskf_core -j1
ctest --test-dir build/ieskf_core --output-on-failure
```

## 运行

### RoboSense Airy 实时数据

先启动 RoboSense 驱动，确认 LiDAR 和 IMU 话题与配置一致，再启动算法与 RViz：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
source install/setup.bash

ros2 launch ieskf_voxel_lio Mapping.launch.py rviz:=true
```

默认配置为 [`config/RobosenseAiry.yaml`](config/RobosenseAiry.yaml)，订阅 `/rslidar_points` 和 `/rslidar_imu_data`。接入前需核实 IMU 单位、LiDAR 逐点时间和雷达—IMU 外参。默认 `imu.accel_scale: 1.0`；若驱动输出加速度单位为 g，应设为 `9.81`。

### 回放 RoboSense mapping 数据集

终端 1 启动节点和 RViz：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
source install/setup.bash

ros2 launch ieskf_voxel_lio Mapping.launch.py \
  config:="$(pwd)/src/IESKF-Voxel-LIO/config/MappingDataset.yaml" \
  use_sim_time:=true rviz:=true
```

等待节点启动后，在终端 2 播放：

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/ros/dataset/robosense/mapping --clock
```

此数据集的 IMU 加速度单位为 g；[`MappingDataset.yaml`](config/MappingDataset.yaml) 设置了 `accel_scale: 9.81` 和 reliable 输入。

### 回放 Livox Mid-360 jungle_challenge

终端 1：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
source install/setup.bash

ros2 launch ieskf_voxel_lio Mapping.launch.py \
  config:="$(pwd)/src/IESKF-Voxel-LIO/config/Mid360Jungle.yaml" \
  use_sim_time:=true rviz:=true
```

终端 2：

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
ros2 bag play /home/ros/dataset/mid360/jungle_challenge --clock
```

该数据集通过 `/livox/lidar` 发布 `livox_ros_driver2/msg/CustomMsg`，通过 `/livox/imu` 发布 IMU。[`Mid360Jungle.yaml`](config/Mid360Jungle.yaml) 中的单位外参是临时假设，应替换为实际标定值。数据开头在运动，约 21～24 秒附近出现静止窗口；初始化之前不输出有效轨迹。

## 雷达输入与配置

配置文件见 [`config/`](config/)。切换设备时，除雷达类型外，还需核对话题、QoS、逐点时间、IMU 单位和外参。

| `preprocess.lidar_type` | 雷达与消息 | 逐点时间字段 |
| --- | --- | --- |
| `1` | RoboSense `PointCloud2` | `timestamp` |
| `2` | Ouster `PointCloud2` | `t`，`UINT32` |
| `3` | Velodyne `PointCloud2` | `time`，`FLOAT32` |
| `4` | Hesai `PointCloud2` | `timestamp` |
| `5` | Livox `PointCloud2` | `timestamp` |
| `5` + `livox_custom_msg: true` | Livox `CustomMsg` | 整数纳秒 `timebase + offset_time` |

ROS 输入层按类型分发，并根据 `PointCloud2` 字段偏移、类型、字节序和行填充读取数据。当前测试过 RoboSense Airy 数据集和 Livox Mid-360 `CustomMsg` 数据集；Ouster、Velodyne、Hesai 仅通过模拟字段测试，具体驱动格式仍需验证。没有逐点时间的 XYZI 消息不受支持。

`PointCloud2` 的 `time_unit`、`time_reference` 和 `header_stamp_reference` 必须与实际驱动一致。常见 Ouster `t` 为相对帧起始的纳秒，常见 Velodyne `time` 为相对帧起始的秒；Livox `CustomMsg` 不使用这些 `PointCloud2` 时间设置。`common.lidar_time_offset_sec` 用于修正 LiDAR 与 IMU 的时间偏移。旧版配置中的字符串 `lidar_type: robosense_airy` 应改为整数 `1`，不再使用 `time_field`。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `common.lidar_topic` / `imu_topic` | 输入话题 |
| `common.reliable_input` | reliable 数据源设为 `true`；best-effort 传感器通常设为 `false` |
| `preprocess.min_range` / `max_range` | 点云有效距离范围，单位 m |
| `preprocess.point_stride` / `scan_voxel_size` | 抽点间隔与去畸变后降采样尺寸 |
| `imu.accel_scale` | 输入加速度转换为 m/s² 的倍率 |
| `imu.init_*` | 静止初始化窗口与阈值 |
| `extrinsic.rotation` / `translation` | LiDAR 到 IMU 的旋转和平移，`p_I = R_IL p_L + t_IL` |
| `ieskf.max_iterations` / `measurement_model` | 最大迭代次数与观测模型 |
| `voxel_map.*` | 体素大小、细分、平面拟合和局部地图范围 |
| `publish.*` | 输出开关、地图周期、平面颜色和轨迹容量 |

完整默认值和单位请查看对应 YAML。`common.world_frame`、`body_frame`、`lidar_frame` 决定坐标系名称；修改世界坐标系后，应同步修改 RViz 的 Fixed Frame。

## ROS 输出与 RViz

以下话题为相对于节点命名空间的名称；默认无命名空间时，以 `/` 开头访问：

| 话题 | 类型 | 内容 |
| --- | --- | --- |
| `odometry` | `nav_msgs/msg/Odometry` | 扫描帧末 IMU 位姿、速度与协方差 |
| `path` | `nav_msgs/msg/Path` | 位姿轨迹 |
| `cloud_deskewed` | `sensor_msgs/msg/PointCloud2` | 帧末 LiDAR 坐标系中的去畸变点云 |
| `cloud_world` | `sensor_msgs/msg/PointCloud2` | 世界坐标系中的当前帧点云 |
| `voxel_planes` | `visualization_msgs/msg/MarkerArray` | 局部地图体素平面 |
| `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 初始化、匹配、耗时和缓冲区状态 |

默认 RViz 固定坐标系为 `odom`。在左侧启用 **Voxel planes** 可显示平面；配置文件需启用 `publish.voxel_planes`。平面颜色依据法向量协方差 3×3 子矩阵的迹从蓝到红变化：蓝色表示估计不确定性较小，红色表示较大。`publish.plane_color_max_normal_cov_trace` 控制色标上限，`publish.plane_color_gamma` 调整低值区间的显示；颜色不直接表示真实定位误差。

## 数据集回放记录

本机使用 i7-10750H、1 倍速回放；耗时为算法核心处理时间，不包含 ROS 输入解析、排队和发布。以下是功能回放记录，并非精度基准。

| 数据集与模型 | 输入帧 | 里程计帧 | 处理失败 / 缓冲区复位 | 平均 / P95 核心耗时 |
| --- | ---: | ---: | ---: | ---: |
| RoboSense mapping，constant（历史记录） | 984 | 973 | 0 / 0 | 13.87 / 17.27 ms |
| RoboSense mapping，probabilistic（历史记录） | 984 | 973 | 0 / 0 | 15.47 / 18.69 ms |
| Mid-360 jungle_challenge，constant | 1199 | 979 | 0 / 0 | 38.33 / 56.70 ms |

Mid-360 输出轨迹跨度约 97.8 秒，开始阶段等待静止初始化；统计到 1 帧雷达丢弃。978 帧观测更新中，55 帧达到收敛条件，923 帧达到 4 次迭代上限，需要进一步分析。该次测试采用未标定的单位外参，数据没有用于本次评估的真值，不能据此得出定位精度或闭环误差结论。

本机结果分别保存在工作区 `test_results/mapping_report.md` 和 `test_results/mid360_jungle_20261124/`；后者包含 `summary.json`、`diagnostics.csv`、`trajectory.tum` 和按帧采样的点云 PLY。采样点云不等于完整地图。

可用 [`tests/ReplayDataset.py`](tests/ReplayDataset.py) 在独立 ROS domain 内回放并保存统计；输出目录必须不存在。例如运行 Mid-360：

```bash
cd /home/ros/ros_ws/fast-lio2-ws
source /opt/ros/humble/setup.bash
source /home/ros/ros_ws/livox_ros/install/setup.bash
source install/setup.bash

/usr/bin/python3 src/IESKF-Voxel-LIO/tests/ReplayDataset.py \
  --node install/ieskf_voxel_lio/lib/ieskf_voxel_lio/lio_node \
  --bag /home/ros/dataset/mid360/jungle_challenge \
  --config src/IESKF-Voxel-LIO/config/Mid360Jungle.yaml \
  --output test_results/my_mid360_run --min-coverage 0.75
```

这里将覆盖率阈值从默认的 0.9 调为 0.75，以容纳初始化等待。脚本的 `passed` 检查覆盖率、修正比例和处理失败数；达到迭代上限的有效更新也计入修正，因此 `passed` 不表示全部帧收敛或定位精度达标。

## 项目结构

```text
IESKF-Voxel-LIO/
├── include/lio/         # 核心与 ROS 接口头文件
├── src/                 # IESKF、IMU、预处理、地图和 ROS 节点
├── config/              # 雷达及数据集配置
├── launch/              # ROS 2 启动文件
├── rviz/                # RViz 配置
├── docs/images/         # 可视化演示
└── tests/               # 单元测试、ROS 冒烟和数据集回放脚本
```

## 已知限制

- 初始化依赖满足阈值的静止 IMU 数据；运动数据开头无法立即建立可靠初始姿态。
- 相邻扫描重叠不超过 1 ms 时裁去旧时刻点，更大重叠拒绝；IMU 必须覆盖扫描时间。
- 地图和状态使用 `float32`，远离世界原点时精度会下降；当前不提供大地坐标精度保证或自动重定位。
- 平面协方差采用独立点近似，未完整建模同帧地图点和历史状态的相关性。
- 当前输出当前帧世界点云与局部平面，未提供完整历史全局点云发布或导出。
- 真实设备外参、不同驱动版本、长时间运行和轨迹精度仍需按具体场景验证。
