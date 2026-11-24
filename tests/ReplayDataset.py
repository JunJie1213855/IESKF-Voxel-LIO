#!/usr/bin/env python3
"""回放完整 rosbag，保存轨迹、诊断、采样点云和可复查的性能统计。"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml


def stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--min-coverage', type=float, default=0.9,
                        help='最少里程计帧占全部雷达帧的比例；默认 0.9')
    parser.add_argument('--model', choices=['constant', 'probabilistic'])
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ValueError('rate must be finite and positive')
    if not math.isfinite(args.min_coverage) or not 0 <= args.min_coverage <= 1:
        raise ValueError('min-coverage must be between 0 and 1')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 40)
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from rosbag2_interfaces.srv import Resume

    metadata = yaml.safe_load((Path(args.bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    duration = metadata['duration']['nanoseconds'] * 1e-9
    bag_counts = {topic['topic_metadata']['name']: topic['message_count']
                  for topic in metadata['topics_with_message_count']}
    config = yaml.safe_load(Path(args.config).read_text())
    params = config['/**']['ros__parameters']
    params['use_sim_time'] = True
    # 隔离 ROS domain；仍关闭 TF，防止离线评估产生额外 TF 发布者。
    params['publish']['tf'] = False
    if args.model:
        params['ieskf']['measurement_model'] = args.model
    config_path = output / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    trajectory = []
    diagnostics = []
    cloud_samples = []
    cloud_count = 0
    rss = []
    rclpy.init()
    driver = rclpy.create_node('lio_dataset_evaluator')

    def odometry(message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        t = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if not all(math.isfinite(v) for v in values + list(message.pose.covariance)):
            raise RuntimeError('nonfinite odometry or covariance')
        trajectory.append([t] + values)

    def diagnostic(message):
        for status in message.status:
            row = {value.key: value.value for value in status.values}
            row['timestamp_ns'] = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            row['status'] = status.message
            diagnostics.append(row)

    def cloud(message):
        nonlocal cloud_count
        cloud_count += 1
        # 只采样每十帧，减少评估器自身开销；该点云不是算法完整地图导出。
        if cloud_count % 10 or not message.width:
            return
        fields = {field.name: field for field in message.fields}
        xyz = [np.ndarray((message.height, message.width), dtype='<f4',
                          buffer=message.data, offset=fields[axis].offset,
                          strides=(message.row_step, message.point_step)).ravel()
               for axis in ('x', 'y', 'z')]
        cloud_samples.append(np.column_stack(xyz).copy())

    driver.create_subscription(Odometry, '/lio_dataset/odometry', odometry, 100)
    driver.create_subscription(DiagnosticArray, '/lio_dataset/diagnostics', diagnostic, 100)
    driver.create_subscription(PointCloud2, '/lio_dataset/cloud_world', cloud, qos_profile_sensor_data)
    bag_process = None
    begin = time.monotonic()
    with (output / 'node.log').open('w') as node_log, (output / 'playback.log').open('w') as bag_log:
        node = subprocess.Popen([
            str(Path(args.node).resolve()), '--ros-args', '--params-file', str(config_path),
            '-r', '__ns:=/lio_dataset',
        ], stdout=node_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while driver.count_publishers('/lio_dataset/diagnostics') == 0:
                if node.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('LIO node startup failed; inspect node.log')
                rclpy.spin_once(driver, timeout_sec=0.02)
            # 给 DDS discovery 留出时间，避免数据集最初的静止窗口被启动时序吞掉。
            discovery_end = time.monotonic() + 1
            while time.monotonic() < discovery_end:
                rclpy.spin_once(driver, timeout_sec=0.02)
            bag_process = subprocess.Popen([
                'ros2', 'bag', 'play', str(Path(args.bag).resolve()), '--clock', '100',
                '--rate', str(args.rate), '--disable-keyboard-controls', '--start-paused',
            ], stdout=bag_log, stderr=subprocess.STDOUT)
            # 先暂停建立所有 bag 发布者，再等待 discovery，避免丢掉最初静止窗口。
            resume = driver.create_client(Resume, '/rosbag2_player/resume')
            discovery_deadline = time.monotonic() + 10
            while not resume.wait_for_service(timeout_sec=0.05):
                rclpy.spin_once(driver, timeout_sec=0.02)
                if bag_process.poll() is not None or time.monotonic() > discovery_deadline:
                    raise RuntimeError('paused bag player did not expose resume service')
            discovery_end = time.monotonic() + 2
            while time.monotonic() < discovery_end:
                rclpy.spin_once(driver, timeout_sec=0.02)
            future = resume.call_async(Resume.Request())
            rclpy.spin_until_future_complete(driver, future, timeout_sec=5)
            if not future.done() or future.result() is None:
                raise RuntimeError('failed to resume bag playback')
            last_progress = time.monotonic()
            deadline = time.monotonic() + duration / args.rate + 60
            while bag_process.poll() is None:
                rclpy.spin_once(driver, timeout_sec=0.02)
                if node.poll() is not None:
                    raise RuntimeError('LIO node exited during playback')
                if time.monotonic() > deadline:
                    raise RuntimeError('playback timeout')
                if time.monotonic() - last_progress >= 15:
                    last_progress = time.monotonic()
                    print(f'progress: odometry={len(trajectory)}, diagnostics={len(diagnostics)}', flush=True)
                status_path = Path(f'/proc/{node.pid}/status')
                if status_path.exists():
                    for line in status_path.read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            rss.append(int(line.split()[1]) / 1024)
            if bag_process.returncode:
                raise RuntimeError('rosbag playback failed; inspect playback.log')
            # 播放器结束后排空估计队列和 DDS 接收队列。
            flush_end = time.monotonic() + 3
            while time.monotonic() < flush_end:
                rclpy.spin_once(driver, timeout_sec=0.02)
            if not trajectory:
                raise RuntimeError('no odometry output; inspect initialization diagnostics')
        finally:
            stop(bag_process)
            stop(node)
            driver.destroy_node()
            rclpy.shutdown()
    with (output / 'trajectory.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows(trajectory)
    with (output / 'trajectory.tum').open('w') as file:
        for row in trajectory:
            seconds, ns = divmod(row[0], 1000000000)
            file.write(f'{seconds}.{ns:09d} ' + ' '.join(f'{v:.9g}' for v in row[1:]) + '\n')
    if diagnostics:
        with (output / 'diagnostics.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    positions = np.array([row[1:4] for row in trajectory])
    initialized = [row for row in diagnostics if int(row.get('initialized', 0))]
    corrected = [row for row in initialized if row['status'] in ('converged', 'iteration limit')]
    summary = {
        'bag': str(Path(args.bag).resolve()), 'bag_duration_sec': duration,
        'bag_messages': bag_counts, 'playback_rate': args.rate,
        'measurement_model': params['ieskf']['measurement_model'],
        'startup_mode': 'start_paused_then_resume_after_discovery',
        'odometry_count': len(trajectory), 'diagnostic_count': len(diagnostics),
        'corrected_frames': len(corrected), 'initialized_frames': len(initialized),
        'trajectory_duration_sec': (trajectory[-1][0] - trajectory[0][0]) * 1e-9,
        'path_length_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        'start_to_end_distance_m': float(np.linalg.norm(positions[-1] - positions[0])),
        'final_position_m': positions[-1].tolist(),
        'bounding_box_m': {'min': positions.min(axis=0).tolist(), 'max': positions.max(axis=0).tolist()},
        'max_rss_mib': max(rss, default=0),
        'wall_time_sec': time.monotonic() - begin,
        'statuses': {status: sum(row['status'] == status for row in diagnostics)
                     for status in sorted({row['status'] for row in diagnostics})},
        'note': 'No ground truth: path length and start/end displacement are not trajectory accuracy metrics.',
    }
    for key in ('failures', 'imu_dropped', 'lidar_dropped', 'buffer_resets', 'map_voxels'):
        summary['max_' + key] = max((int(row[key]) for row in diagnostics), default=0)
    for key in ('total_ms', 'propagation_ms', 'deskew_ms', 'update_ms', 'mapping_ms',
                'effective_points', 'mean_abs_residual'):
        values = np.array([float(row[key]) for row in initialized])
        summary[key] = {'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
                        'p95': float(np.percentile(values, 95)), 'max': float(values.max())}
    summary['trimmed_overlap_points'] = sum(int(row.get('trimmed_overlap_points', 0)) for row in diagnostics)
    expected_scans = bag_counts.get(params['common']['lidar_topic'], 0)
    summary['min_coverage'] = args.min_coverage
    summary['passed'] = (expected_scans > 0 and len(trajectory) >= args.min_coverage * expected_scans and
                         summary['max_failures'] == 0 and
                         len(corrected) >= 0.95 * len(initialized))
    summary['hardware'] = next((line.split(':', 1)[1].strip()
                                for line in Path('/proc/cpuinfo').read_text().splitlines()
                                if line.startswith('model name')), 'unknown')
    if cloud_samples:
        points = np.concatenate(cloud_samples)
        points = points[np.isfinite(points).all(axis=1)]
        _, selected = np.unique(np.floor(points / 0.15).astype(np.int64), axis=0, return_index=True)
        points = points[selected].astype('<f4')
        with (output / 'sampled_cloud.ply').open('wb') as file:
            file.write(('ply\nformat binary_little_endian 1.0\n'
                        f'element vertex {len(points)}\nproperty float x\nproperty float y\n'
                        'property float z\nend_header\n').encode())
            file.write(points.tobytes())
        summary['sampled_cloud_points'] = len(points)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary['passed']:
        raise RuntimeError('dataset acceptance checks failed; see summary.json')


if __name__ == '__main__':
    main()
