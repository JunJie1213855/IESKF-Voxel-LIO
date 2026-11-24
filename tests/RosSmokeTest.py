#!/usr/bin/env python3
"""在隔离话题上验证完整 ROS 输入、初始化、建图和观测更新链路。"""
import argparse
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 40)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from nav_msgs.msg import Odometry
    from diagnostic_msgs.msg import DiagnosticArray
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import Marker, MarkerArray

    config = yaml.safe_load(Path(args.config).read_text())
    params = config['/**']['ros__parameters']
    params['common']['lidar_topic'] = '/lio_smoke/input_lidar'
    params['common']['imu_topic'] = '/lio_smoke/input_imu'
    params['extrinsic']['rotation'] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    params['imu']['init_duration_sec'] = 0.2
    params['imu']['init_min_samples'] = 30
    params['preprocess']['point_stride'] = 1
    params['preprocess']['scan_voxel_size'] = 0.25
    params['publish']['tf'] = True
    params['publish']['voxel_planes'] = True
    odometry = []
    diagnostics = []
    dynamic_tf = []
    static_tf = []
    plane_messages = []
    rclpy.init()
    driver = rclpy.create_node('lio_smoke_driver')
    imu_pub = driver.create_publisher(Imu, '/lio_smoke/input_imu', qos_profile_sensor_data)
    cloud_pub = driver.create_publisher(PointCloud2, '/lio_smoke/input_lidar', qos_profile_sensor_data)
    subscriptions = [
        driver.create_subscription(Odometry, '/lio_smoke/odometry', odometry.append, 100),
        driver.create_subscription(DiagnosticArray, '/lio_smoke/diagnostics', diagnostics.append, 100),
        driver.create_subscription(TFMessage, '/tf', dynamic_tf.append, 100),
        driver.create_subscription(MarkerArray, '/lio_smoke/voxel_planes', plane_messages.append, 10),
        driver.create_subscription(TFMessage, '/tf_static', static_tf.append,
                                   QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
    ]
    del subscriptions  # Node 本身保留订阅对象所有权。
    with tempfile.TemporaryDirectory(prefix='lio_smoke_') as directory:
        config_path = Path(directory) / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        log_path = Path(directory) / 'node.log'
        with log_path.open('w') as log:
            process = subprocess.Popen([
                str(Path(args.node).resolve()), '--ros-args', '--params-file', str(config_path),
                '-r', '__ns:=/lio_smoke',
            ], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 8
                while driver.count_subscribers('/lio_smoke/input_imu') == 0:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('LIO node did not subscribe to synthetic input')
                    rclpy.spin_once(driver, timeout_sec=0.02)
                for frame in range(30):
                    for k in range(0 if frame == 0 else 1, 21):
                        stamp = 1000000000000 + frame * 100000000 + k * 5000000
                        imu = Imu()
                        imu.header.stamp.sec, imu.header.stamp.nanosec = divmod(stamp, 1000000000)
                        imu.linear_acceleration.z = 9.81
                        imu_pub.publish(imu)
                        rclpy.spin_once(driver, timeout_sec=0.001)
                        time.sleep(0.001)
                    cloud = PointCloud2()
                    start = 1000000000000 + frame * 100000000
                    cloud.header.stamp.sec, cloud.header.stamp.nanosec = divmod(start, 1000000000)
                    cloud.header.frame_id = 'synthetic_lidar'
                    cloud.fields = [PointField(name=name, offset=offset, datatype=kind, count=1)
                                    for name, offset, kind in [
                                        ('x', 0, PointField.FLOAT32), ('y', 4, PointField.FLOAT32),
                                        ('z', 8, PointField.FLOAT32), ('intensity', 12, PointField.FLOAT32),
                                        ('timestamp', 16, PointField.FLOAT64)]]
                    data = bytearray()
                    index = 0
                    for plane in range(3):
                        for i in range(-7, 8):
                            for j in range(-7, 8):
                                if plane == 0:
                                    xyz = (i * 0.4, j * 0.4, -1.0)
                                elif plane == 1:
                                    xyz = (5.0, i * 0.4, 1 + j * 0.25)
                                else:
                                    xyz = (i * 0.4, 5.0, 1 + j * 0.25)
                                point_time = start * 1e-9 + (index % 101) * 0.001
                                data.extend(struct.pack('<ffffd', *xyz, 10.0, point_time))
                                index += 1
                    cloud.width = index
                    cloud.height = 1
                    cloud.point_step = 24
                    cloud.row_step = index * 24
                    cloud.data = bytes(data)
                    cloud.is_dense = True
                    cloud_pub.publish(cloud)
                    for _ in range(5):
                        rclpy.spin_once(driver, timeout_sec=0.01)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    rclpy.spin_once(driver, timeout_sec=0.02)
                values = [{value.key: value.value for value in status.values}
                          for message in diagnostics for status in message.status]
                plane_colors = {(round(marker.color.r, 4), round(marker.color.g, 4),
                                 round(marker.color.b, 4))
                                for msg in plane_messages for marker in msg.markers
                                if marker.action == Marker.ADD}
                summary = {
                    'odometry_messages': len(odometry),
                    'diagnostic_messages': len(diagnostics),
                    'dynamic_tf_messages': len(dynamic_tf),
                    'static_tf_messages': len(static_tf),
                    'plane_messages': len(plane_messages),
                    'distinct_plane_colors': len(plane_colors),
                    'max_effective_points': max((int(v['effective_points']) for v in values), default=0),
                    'max_failures': max((int(v['failures']) for v in values), default=0),
                    'final_position': ([odometry[-1].pose.pose.position.x,
                                        odometry[-1].pose.pose.position.y,
                                        odometry[-1].pose.pose.position.z] if odometry else None),
                }
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                if (len(odometry) < 10 or summary['max_effective_points'] < 30 or
                        summary['max_failures'] or len(dynamic_tf) < 10 or not static_tf or
                        len(plane_colors) < 2):
                    raise RuntimeError('synthetic ROS pipeline did not pass acceptance checks')
                if sum(x * x for x in summary['final_position']) > 0.01:
                    raise RuntimeError('stationary synthetic trajectory drift exceeds 0.1 m')
            except Exception:
                print(log_path.read_text())
                raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                driver.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    main()
