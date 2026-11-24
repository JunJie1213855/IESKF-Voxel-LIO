#!/usr/bin/env python3
"""生成两种观测模型的回放对照图；不将两者差异解释为真值误差。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    with (directory / 'trajectory.csv').open() as file:
        trajectory = list(csv.DictReader(file))
    positions = np.array([[float(row[axis]) for axis in ('x', 'y', 'z')]
                          for row in trajectory])
    origin = int(trajectory[0]['timestamp_ns'])
    times = np.array([(int(row['timestamp_ns']) - origin) * 1e-9 for row in trajectory])
    with (directory / 'diagnostics.csv').open() as file:
        diagnostics = [row for row in csv.DictReader(file) if int(row['initialized'])]
    diagnostic_times = np.array([(int(row['timestamp_ns']) - origin) * 1e-9
                                 for row in diagnostics])
    cloud_path = directory / 'sampled_cloud.ply'
    cloud = None
    if cloud_path.exists():
        with cloud_path.open('rb') as file:
            while file.readline().strip() != b'end_header':
                pass
            cloud = np.frombuffer(file.read(), dtype='<f4').reshape(-1, 3)
    return summary, positions, times, diagnostics, diagnostic_times, cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constant', required=True)
    parser.add_argument('--probabilistic', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = [read_run(args.constant), read_run(args.probabilistic)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ['#1764a0', '#d77818']
    labels = ['Constant R', 'Probabilistic R']
    if runs[0][-1] is not None:
        cloud = runs[0][-1]
        axes[0, 0].scatter(cloud[:, 0], cloud[:, 1], c='#aab4bc', s=1,
                           alpha=0.25, rasterized=True, label='Sampled registered points')
    for run, color, label in zip(runs, colors, labels):
        summary, positions, times, rows, dt, _ = run
        axes[0, 0].plot(positions[:, 0], positions[:, 1], color=color, lw=1.3, label=label)
        axes[0, 0].scatter(*positions[-1, :2], color=color, marker='x', s=35)
        axes[0, 1].plot(times, positions[:, 2], color=color, label=label)
        axes[1, 0].plot(dt, [float(row['total_ms']) for row in rows], color=color,
                        alpha=0.7, lw=0.7,
                        label=f"{label}: P95 {summary['total_ms']['p95']:.1f} ms")
        axes[1, 1].plot(dt, [float(row['mean_abs_residual']) for row in rows],
                        color=color, lw=0.7, alpha=0.8, label=label)
    axes[0, 0].scatter(0, 0, marker='o', s=35, color='#292f36', label='Initial origin')
    axes[0, 0].set(title='Estimated trajectory and sampled registered points', xlabel='World X (m)', ylabel='World Y (m)')
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 1].set(title='Estimated height', xlabel='Time after first estimate (s)', ylabel='World Z (m)')
    axes[1, 0].axhline(100, color='#ab3a3a', ls='--', lw=1, label='10 Hz frame period')
    axes[1, 0].set(title='Core processing time (publishing excluded)', xlabel='Time after first estimate (s)', ylabel='Time (ms)')
    axes[1, 1].set(title='Mean absolute point-to-plane residual', xlabel='Time after first estimate (s)', ylabel='Residual (m)')
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle('IESKF-Voxel-LIO | mapping dataset | full 1x replay', fontsize=13)
    fig.text(0.5, 0.015, 'No ground truth provided. Differences between runs are not ATE/RPE. Point cloud is sampled every ten output frames.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
