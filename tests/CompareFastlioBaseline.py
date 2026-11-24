#!/usr/bin/env python3
"""Replay an isolated FAST-LIO2 reference and compare same-time body poses."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


def stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def read_trajectory(path):
    with Path(path).open() as file:
        rows = list(csv.DictReader(file))
    times = np.array([int(row['timestamp_ns']) for row in rows], dtype=np.int64)
    poses = np.array([[float(row[k]) for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] for row in rows])
    return times, poses


def compare(reference_path, candidate_path, output):
    ref_t, ref = read_trajectory(reference_path)
    cand_t, cand = read_trajectory(candidate_path)
    if len(ref) < 2 or len(cand) < 2 or np.any(np.diff(ref_t) <= 0) or np.any(np.diff(cand_t) <= 0):
        raise ValueError('trajectories must have at least two poses with strictly increasing timestamps')
    # 只插值相邻两帧之间的正常时间范围，避免跨缺帧伪造比较结果。
    indices = np.searchsorted(ref_t, cand_t)
    valid = (indices > 0) & (indices < len(ref_t))
    selected = np.flatnonzero(valid)
    selected = selected[(ref_t[indices[selected]] - ref_t[indices[selected] - 1]) <= 200_000_000]
    if len(selected) < 10:
        raise ValueError('insufficient overlapping timestamps')
    times = cand_t[selected]
    start = min(ref_t[0], times[0])
    ref_s = (ref_t - start).astype(np.float64) * 1e-9
    query = (times - start).astype(np.float64) * 1e-9
    ref_pos = np.column_stack([np.interp(query, ref_s, ref[:, k]) for k in range(3)])
    ref_rot = Slerp(ref_s, Rotation.from_quat(ref[:, 3:7]))(query)
    cand_pos = cand[selected, :3]
    cand_rot = Rotation.from_quat(cand[selected, 3:7])
    # 使用第一个共同时间的完整 IMU 位姿一次性对齐，消除世界系初始航向和原点的差别。
    align = ref_rot[0] * cand_rot[0].inv()
    aligned = align.apply(cand_pos - cand_pos[0]) + ref_pos[0]
    aligned_rot = align * cand_rot
    delta = aligned - ref_pos
    distance = np.linalg.norm(delta, axis=1)
    angle = (ref_rot.inv() * aligned_rot).magnitude() * 180 / np.pi
    metrics = {
        'alignment': 'single full SE(3) transform from the first matched IMU pose; no scale fitting',
        'matched_poses': len(times),
        'overlap_sec': float((times[-1] - times[0]) * 1e-9),
        'position_difference_m': {'rmse': float(np.sqrt(np.mean(distance ** 2))),
                                  'median': float(np.median(distance)),
                                  'p95': float(np.percentile(distance, 95)),
                                  'max': float(distance.max()), 'last': float(distance[-1])},
        'rotation_difference_deg': {'rmse': float(np.sqrt(np.mean(angle ** 2))),
                                    'p95': float(np.percentile(angle, 95)),
                                    'max': float(angle.max()), 'last': float(angle[-1])},
        'interpretation': 'Inter-algorithm differences; FAST-LIO2 is not ground truth and this is not ATE/RPE.',
    }
    with (output / 'matched_poses.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ns', 'reference_x', 'reference_y', 'reference_z',
                         'candidate_aligned_x', 'candidate_aligned_y', 'candidate_aligned_z',
                         'position_difference_m', 'rotation_difference_deg'])
        for t, p, a, d, theta in zip(times, ref_pos, aligned, distance, angle):
            writer.writerow([t, *p, *a, d, theta])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rate', type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0:
        raise ValueError('rate must be positive and finite')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 40)
    import rclpy
    from nav_msgs.msg import Odometry
    from rosbag2_interfaces.srv import Resume

    config = yaml.safe_load(Path(args.config).read_text())
    p = config['/**']['ros__parameters']
    p['use_sim_time'] = True
    p['feature_extract_enable'] = False
    p['runtime_pos_log_enable'] = True
    p['max_iteration'] = 4
    p['pcd_save']['pcd_save_en'] = False
    p['publish'].update({'path_en': False, 'scan_publish_en': False,
                         'map_en': False, 'effect_map_en': False})
    config_path = output / 'baseline_config.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    metadata = yaml.safe_load((Path(args.bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    duration = metadata['duration']['nanoseconds'] * 1e-9
    samples, covariance, rss = [], [], []
    rclpy.init()
    observer = rclpy.create_node('fastlio_baseline_observer')

    def odometry(msg):
        q = msg.pose.pose.orientation
        x = msg.pose.pose.position
        values = [x.x, x.y, x.z, q.x, q.y, q.z, q.w]
        t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if all(math.isfinite(v) for v in values):
            samples.append([t, *values])
            covariance.append([t, *msg.pose.covariance])

    observer.create_subscription(Odometry, '/Odometry', odometry, 100)
    bag = None
    with (output / 'node.log').open('w') as node_log, (output / 'playback.log').open('w') as bag_log:
        node = subprocess.Popen([str(Path(args.node).resolve()), '--ros-args', '--params-file', str(config_path)],
                                stdout=node_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while observer.count_publishers('/Odometry') == 0:
                if node.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('baseline node failed to start; see node.log')
                rclpy.spin_once(observer, timeout_sec=.02)
            bag = subprocess.Popen(['ros2', 'bag', 'play', str(Path(args.bag).resolve()), '--clock', '100',
                                    '--rate', str(args.rate), '--disable-keyboard-controls', '--start-paused'],
                                   stdout=bag_log, stderr=subprocess.STDOUT)
            resume = observer.create_client(Resume, '/rosbag2_player/resume')
            deadline = time.monotonic() + 15
            while not resume.wait_for_service(timeout_sec=.05):
                rclpy.spin_once(observer, timeout_sec=.02)
                if bag.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('bag service unavailable')
            end = time.monotonic() + 2
            while time.monotonic() < end:
                rclpy.spin_once(observer, timeout_sec=.02)
            future = resume.call_async(Resume.Request())
            rclpy.spin_until_future_complete(observer, future, timeout_sec=5)
            if not future.done() or future.result() is None:
                raise RuntimeError('bag resume failed')
            deadline = time.monotonic() + duration / args.rate + 60
            next_progress = time.monotonic() + 15
            while bag.poll() is None:
                rclpy.spin_once(observer, timeout_sec=.02)
                if node.poll() is not None:
                    raise RuntimeError('baseline exited during replay; see node.log')
                if time.monotonic() > deadline:
                    raise RuntimeError('bag replay timeout')
                if time.monotonic() >= next_progress:
                    print('FAST-LIO2 odometry:', len(samples), flush=True)
                    next_progress = time.monotonic() + 15
                status = Path(f'/proc/{node.pid}/status')
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            rss.append(int(line.split()[1]) / 1024)
            if bag.returncode:
                raise RuntimeError('bag replay failed; see playback.log')
            end = time.monotonic() + 4
            while time.monotonic() < end:
                rclpy.spin_once(observer, timeout_sec=.02)
        finally:
            stop(bag)
            stop(node)
            observer.destroy_node()
            rclpy.shutdown()
    with (output / 'trajectory.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows(samples)
    with (output / 'published_covariance.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ns'] + [f'p{r}{c}' for r in range(6) for c in range(6)])
        writer.writerows(covariance)
    (output / 'trajectory.tum').write_text(''.join(f'{t // 1000000000}.{t % 1000000000:09d} ' +
        ' '.join(f'{v:.9g}' for v in pose) + '\n' for t, *pose in samples))
    summary = {'odometry_count': len(samples), 'max_rss_mib': max(rss, default=0),
               'rate': args.rate, 'binary_sha256': hashlib.sha256(Path(args.node).read_bytes()).hexdigest(),
               'published_covariance_caveat': 'Reference publishes the odometry message before copying the latest covariance; published covariance is stale.'}
    try:
        summary['comparison'] = compare(output / 'trajectory.csv', args.candidate, output)
    except ValueError as exc:
        summary['comparison_error'] = str(exc)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if 'comparison_error' in summary:
        raise RuntimeError(summary['comparison_error'])


if __name__ == '__main__':
    main()
