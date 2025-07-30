#!/usr/bin/env python3

# Added by Jaemoon Park, Tommoro Robotics

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    cfg_arg = DeclareLaunchArgument(
        'cfg', default_value='poses/desk_pose.yaml',
        description='Relative YAML path under config/<model>/'
    )
    cfg = LaunchConfiguration('cfg')

    model = 'ffw_bg2_rev4_follower'
    cfg_path = PathJoinSubstitution([
        FindPackageShare('ffw_bringup'),
        'config', model, cfg
    ])

    def node(name):
        return Node(
            package='ffw_bringup',
            executable='joint_trajectory_executor',
            name=name,
            parameters=[cfg_path],
            output='screen',
        )

    return LaunchDescription([
        cfg_arg,
        node('arm_l_joint_trajectory_executor'),
        node('arm_r_joint_trajectory_executor'),
        node('head_joint_trajectory_executor'),
        node('lift_joint_trajectory_executor'),
    ])
