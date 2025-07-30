#!/usr/bin/env python3

# Added by Jaemoon Park, Tommoro Robotics


import os, subprocess, sys, textwrap
from ament_index_python.packages import get_package_share_directory

PKG        = 'ffw_bringup'
MODEL_DIR  = 'ffw_bg2_rev4_follower'
POSE_DIR   = 'poses'
TRAJ_DIR   = 'trajectories'
LAUNCH     = 'ffw_bg2_follower_poser.launch.py'

def list_yaml(subdir):
    base = os.path.join(get_package_share_directory(PKG), 'config', MODEL_DIR, subdir)
    return base, sorted(f for f in os.listdir(base) if f.endswith('.yaml'))

def menu(title, items):
    print(f'\n===============================')
    print(f'Select {title.capitalize()}:\n')
    for i, it in enumerate(items, 1):
        print(f'[{i}] {os.path.splitext(it)[0]}')
    print('[0] Return to menu')
    print(f'===============================\n')

    while True:
        try:
            i = int(input('Select Option: '))
            if 0 <= i <= len(items):
                return i
        except ValueError:
            print(f'Wrong Input \'{i}\'...')

def main():
    poses_dir, poses = list_yaml(POSE_DIR)
    trajs_dir, trajs = list_yaml(TRAJ_DIR)

    while True:
        print(textwrap.dedent("""
            ======= AI Worker Poser =======
            
            Select Mode:

            [1] Poses
            [2] Trajectories
            [0] Exit
            ===============================\n"""))
        top = input('Select Option: ')
        if top == '1':
            idx = menu('poses', poses)
            if idx == 0: continue
            rel_path = f'{POSE_DIR}/{poses[idx-1]}'
        elif top == '2':
            idx = menu('trajectories', trajs)
            if idx == 0: continue
            rel_path = f'{TRAJ_DIR}/{trajs[idx-1]}'
        elif top == '0':
            sys.exit(0)
        else:
            continue

        cmd = ['ros2', 'launch', PKG, LAUNCH, f'cfg:={rel_path}']
        print('\n▶', ' '.join(cmd), '\n')
        subprocess.run(cmd)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
