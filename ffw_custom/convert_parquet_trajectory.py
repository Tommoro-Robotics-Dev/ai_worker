#!/usr/bin/env python3

# Added by Jaemoon Park, Tommoro Robotics

import argparse, os, sys, textwrap
from pathlib import Path
import pandas as pd
import yaml

# Joint index tables
ARM_L_JOINTS = [
    'arm_l_joint1','arm_l_joint2','arm_l_joint3','arm_l_joint4',
    'arm_l_joint5','arm_l_joint6','arm_l_joint7','gripper_l_joint1'
]
ARM_L_INDICES = [0,1,2,3,4,5,6,7]

ARM_R_JOINTS = [
    'arm_r_joint1','arm_r_joint2','arm_r_joint3','arm_r_joint4',
    'arm_r_joint5','arm_r_joint6','arm_r_joint7','gripper_r_joint1'
]
ARM_R_INDICES = [8,9,10,11,12,13,14,15]

HEAD_JOINTS = ['head_joint1','head_joint2']
HEAD_INDICES = [16,17]

LIFT_JOINTS = ['lift_joint']
LIFT_INDICES = [18]

EXECUTOR_TABLE = {
    "arm_l_joint_trajectory_executor": (ARM_L_JOINTS, ARM_L_INDICES, "/arm_l_controller/follow_joint_trajectory"),
    "arm_r_joint_trajectory_executor": (ARM_R_JOINTS, ARM_R_INDICES, "/arm_r_controller/follow_joint_trajectory"),
    "head_joint_trajectory_executor":  (HEAD_JOINTS,   HEAD_INDICES, "/head_controller/follow_joint_trajectory"),
    "lift_joint_trajectory_executor":  (LIFT_JOINTS,   LIFT_INDICES, "/lift_controller/follow_joint_trajectory"),
}

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_GOAL_TIMEOUT = 5.0


def flow_list(lst: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in lst) + "]"

def build_block(actions, indices, t_rel):
    lines=[]
    for vec,t in zip(actions,t_rel):
        pos = flow_list([vec[i] for i in indices])
        lines.append(f"- positions: {pos}\n  time_from_start: {t}")
    return textwrap.indent("\n".join(lines), "      ")

def indent_list(joints):
    # 6칸 들여쓰기로 각 joint 이름 출력
    return "\n".join(f"      - {j}" for j in joints)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-p","--parquet",   required=True)
    ap.add_argument("-n","--traj_name", required=True)
    ap.add_argument("-m","--model", default="ffw_bg2_rev4_follower")
    args=ap.parse_args()

    df=pd.read_parquet(args.parquet)
    if not {'action','timestamp'}.issubset(df.columns):
        sys.exit("Parquet must have 'action' and 'timestamp' columns")

    actions=df['action'].tolist()
    ts=df['timestamp'].astype(float).tolist()
    base=ts[0]; t_rel=[round(t-base,6) for t in ts]

    out_dir = Path("/root/ros2_ws/src/ai_worker/ffw_bringup") / "config" / args.model / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path=out_dir/f"{args.traj_name}.yaml"

    sections=[]
    for exe,(joints,idxs,topic) in EXECUTOR_TABLE.items():
        if any(len(a)!=19 for a in actions):
            sys.exit("Each 'action' row must be length 19")

        block=build_block(actions,idxs,t_rel)
        section=f"""{exe}:
  ros__parameters:
    joint_names:
{indent_list(joints)}
    trajectory: |
{block}
    action_topic: "{topic}"
    joint_states_topic: "/joint_states"
    max_goal_attempts: {DEFAULT_MAX_ATTEMPTS}
    goal_timeout: {DEFAULT_GOAL_TIMEOUT}
"""
        sections.append(section.rstrip())

    out_path.write_text("\n\n".join(sections)+"\n")
    print(f"✅ Saved → {out_path}")

if __name__=="__main__":
    main()