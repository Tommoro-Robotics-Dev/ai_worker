#!/usr/bin/env python3
"""
AI Worker Listener Script
Listens on a FIFO for commands to trigger poses or trajectories via ROS2 launch.

Author: Eduardo Jesus Ayerve Cruz, Tommoro Robotics
"""

import os
import sys
import select
import subprocess
from ament_index_python.packages import get_package_share_directory

# Configuration
PKG = 'ffw_bringup'
MODEL_DIR = 'ffw_bg2_rev4_follower'
POSE_DIR = 'poses'
TRAJ_DIR = 'trajectories'
LAUNCH = 'ffw_bg2_follower_poser.launch.py'
FIFO_PATH = '/tmp/ai_worker_cmd'


def setup_fifo(path):
    """Create the FIFO if needed and open it non-blocking for reading."""
    if not os.path.exists(path):
        os.mkfifo(path)
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK)


def load_yaml_list(subdir):
    """Return base path and sorted list of .yaml files in the given subdirectory."""
    base = os.path.join(
        get_package_share_directory(PKG), 'config', MODEL_DIR, subdir
    )
    files = sorted(f for f in os.listdir(base) if f.endswith('.yaml'))
    return base, files


def parse_command(cmd_str, poses, trajs):
    """
    Parses command strings of the form:
      pose <index>
      traj <index>
    Returns ('pose', index) or ('traj', index), or (None, None) on invalid.
    """
    parts = cmd_str.strip().split()
    if len(parts) < 2:
        return None, None
    key = parts[0].lower()
    try:
        idx = int(parts[1])
    except ValueError:
        return None, None
    if key in ('pose', 'p') and 1 <= idx <= len(poses):
        return 'pose', idx
    if key in ('traj', 't', 'trajectory') and 1 <= idx <= len(trajs):
        return 'traj', idx
    return None, None


def main():
    # Setup FIFO listener
    fifo_fd = setup_fifo(FIFO_PATH)

    # Load available YAML configs
    poses_base, poses = load_yaml_list(POSE_DIR)
    trajs_base, trajs = load_yaml_list(TRAJ_DIR)

    print(f"Listening on FIFO {FIFO_PATH} for commands: 'pose <n>' or 'traj <n>'")
    while True:
        # Wait for data on the FIFO
        rlist, _, _ = select.select([fifo_fd], [], [])
        if fifo_fd in rlist:
            try:
                data = os.read(fifo_fd, 128).decode().strip()
            except OSError:
                continue
            if not data:
                continue

            key, idx = parse_command(data, poses, trajs)
            if key == 'pose':
                filename = poses[idx-1]
                rel_path = f'{POSE_DIR}/{filename}'
            elif key == 'traj':
                filename = trajs[idx-1]
                rel_path = f'{TRAJ_DIR}/{filename}'
            else:
                print(f"Ignoring invalid command: {data}")
                continue

            cmd = ['ros2', 'launch', PKG, LAUNCH, f'cfg:={rel_path}']
            print(f"▶ Executing: {' '.join(cmd)}")
            subprocess.Popen(cmd)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting listener.")
        sys.exit(0)
