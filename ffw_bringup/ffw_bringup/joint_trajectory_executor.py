#!/usr/bin/env python3
#
# Copyright 2024 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Sungho Woo
# [2025.07] Modified by: Jaemoon Park, Tommoro Robotics

import math
import sys
import yaml
import numpy as np

import rclpy
from rclpy.node          import Node
from rclpy.action        import ActionClient
from sensor_msgs.msg     import JointState
from action_msgs.msg     import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class JointTrajectoryExecutor(Node):
    """Execute robot joint trajectories in two modes:
       1) step_names   : interpolated motion between named poses
       2) trajectory   : explicit timestamped points
    """

    def __init__(self) -> None:
        super().__init__('joint_trajectory_executor')

        # Declare ROS2 parameters
        self.declare_parameter('joint_names', [''])
        self.declare_parameter('step_names', [''])
        self.declare_parameter('trajectory', '')
        self.declare_parameter('duration', 10.0)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('velocity_tolerance', 0.01)
        self.declare_parameter('max_goal_attempts', 3)
        self.declare_parameter('goal_timeout', 12.0)
        self.declare_parameter('action_topic', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('joint_states_topic', '/joint_states')

        # Load ROS2 parameters
        self.joint_names  = self.get_parameter('joint_names').get_parameter_value().string_array_value
        raw_step_names    = self.get_parameter('step_names').get_parameter_value().string_array_value
        self.step_names   = [s for s in raw_step_names if s]
        self.duration     = self.get_parameter('duration').value
        self.pos_tol      = self.get_parameter('position_tolerance').value
        self.vel_tol      = self.get_parameter('velocity_tolerance').value
        self.max_attempts = self.get_parameter('max_goal_attempts').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.action_topic = self.get_parameter('action_topic').value
        self.jstate_topic = self.get_parameter('joint_states_topic').value

        if not self.joint_names:
            self.get_logger().error('Parameter "joint_names" is required.')
            sys.exit(1)

        self.is_timestamped_mode: bool = False
        self.trajectory: list[JointTrajectoryPoint] = []
        self.positions_list: list[list[float]] = []

        # Parameters for checking timeouts
        self.attempt_count = 0
        self._result_future = None
        self._goal_timer = None

        # Attempt to parse `trajectory`
        raw_traj = self.get_parameter('trajectory').get_parameter_value().string_value
        if raw_traj:
            try:
                parsed = yaml.safe_load(raw_traj)
                if isinstance(parsed, list):
                    for pt in parsed:
                        jtp = JointTrajectoryPoint()
                        jtp.positions = pt['positions']
                        t = float(pt['time_from_start'])
                        jtp.time_from_start.sec     = int(t)
                        jtp.time_from_start.nanosec = int((t % 1.0) * 1e9)
                        if 'velocities' in pt:      jtp.velocities     = pt['velocities']
                        if 'accelerations' in pt:   jtp.accelerations  = pt['accelerations']
                        self.trajectory.append(jtp)
                    self.is_timestamped_mode = True
                    self.get_logger().info(f'Loaded {len(self.trajectory)} timestamped points.')
            except Exception as e:
                self.get_logger().error(f'Failed to parse "trajectory" parameter: {e}')
                sys.exit(1)

        # If no trajectory, fall back to step_names
        if not self.is_timestamped_mode:
            if not self.step_names:
                self.get_logger().error('Provide either "trajectory" or non-empty "step_names".')
                sys.exit(1)

            for step in self.step_names:
                # declare each pose as parameter to fetch numeric array
                self.declare_parameter(step, [0.0]*len(self.joint_names))
                vals = self.get_parameter(step).get_parameter_value().double_array_value
                if len(vals) != len(self.joint_names):
                    self.get_logger().error(
                        f'Step "{step}" length {len(vals)} ≠ joint_names length {len(self.joint_names)}')
                    sys.exit(1)
                self.positions_list.append(vals)
            self.get_logger().info(f'Loaded {len(self.positions_list)} poses ({self.step_names}).')


        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        self.subscription  = self.create_subscription(JointState, self.jstate_topic,
                                                      self.joint_state_cb, 10)

        # runtime state
        self.current_positions  = None
        self.current_velocities = None
        self.goal_handle        = None
        self.reached_target     = False
        self.current_step       = 0
        self.num_points         = 100   # quintic samples

        self.get_logger().info('Waiting for FollowJointTrajectory action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Action server ready.')


    def joint_state_cb(self, msg: JointState) -> None:
        if not set(self.joint_names).issubset(msg.name):
            return

        idx = [msg.name.index(j) for j in self.joint_names]
        self.current_positions  = [msg.position[i]  for i in idx]
        self.current_velocities = [msg.velocity[i]  for i in idx]

        if self.goal_handle is None:
            if self.is_timestamped_mode:
                self._send_full_trajectory()
            else:
                self._send_next_step()

        # check completion for step mode
        if (not self.is_timestamped_mode) and self._step_completed():
            self._advance_step()


    def _send_full_trajectory(self) -> None:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        traj.points      = self.trajectory

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f'Sending timestamped trajectory ({len(traj.points)} points)...')
        future = self.action_client.send_goal_async(goal, feedback_callback=self._fb)
        future.add_done_callback(self._goal_resp_cb)
        # don't set self.goal_handle yet — will in _goal_resp_cb

    # ───────────────────────────── helper: send next step ──
    def _send_next_step(self) -> None:
        if self.current_step >= len(self.positions_list):
            return
        target = self.positions_list[self.current_step]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._make_smooth_traj(self.current_positions, target)

        self.get_logger().info(f'Moving to step {self.current_step} ({self.step_names[self.current_step]})...')
        future = self.action_client.send_goal_async(goal, feedback_callback=self._fb)
        future.add_done_callback(self._goal_resp_cb)

    # ───────────────────────────── step completion check ──
    def _step_completed(self) -> bool:
        target = self.positions_list[self.current_step]
        pos_ok = all(abs(c-t) < self.pos_tol for c, t in zip(self.current_positions, target))
        vel_ok = all(abs(v)   < self.vel_tol for v         in self.current_velocities)
        return pos_ok and vel_ok

    def _advance_step(self) -> None:
        self.get_logger().info(f'Step {self.current_step} completed.')
        self.goal_handle   = None
        self.current_step += 1
        if self.current_step >= len(self.positions_list):
            self.get_logger().info('All steps executed. Shutting down...')
            self._shutdown()
        else:
            # immediately send next
            self._send_next_step()

    # ──────────────────────────────── trajectory build ──
    def _make_smooth_traj(self, start, end) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        times = np.linspace(0.0, self.duration, self.num_points)

        for t in times:
            τ  = t / self.duration
            τ2 = τ*τ; τ3 = τ2*τ; τ4 = τ3*τ; τ5 = τ4*τ

            pos_c = 10*τ3 - 15*τ4 + 6*τ5
            vel_c = (30*τ2 - 60*τ3 + 30*τ4) / self.duration
            acc_c = (60*τ  -180*τ2 +120*τ3) / (self.duration**2)

            p = JointTrajectoryPoint()
            p.time_from_start.sec     = int(t)
            p.time_from_start.nanosec = int((t % 1.0)*1e9)

            for s, e in zip(start, end):
                diff = e - s
                p.positions.append(s + diff*pos_c)
                p.velocities.append(diff*vel_c)
                p.accelerations.append(diff*acc_c)

            traj.points.append(p)
        return traj

    # ────────────────────────────── action callbacks ──
    def _fb(self, fb):
        self.get_logger().debug(f'Feedback positions {fb.feedback.actual.positions[:2]}...')


    def _result_cb(self, future):
        # result = future.result().result
        status = future.result().status     # Value of GoalStatus.msg enum

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Trajectory execution succeeded.')
        else:
            self.get_logger().warn(f'Trajectory finished with status {status}.')

        self._shutdown()


    def _goal_resp_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected by controller.')
            return

        self.get_logger().info('Goal accepted. Waiting for result...')
        self.goal_handle = handle

        # Register callback for final positions
        self._result_future = handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)

    # ───────────────────────────────────────── misc ──
    def _shutdown(self):
        if self.goal_handle:
            self.goal_handle.cancel_goal_async()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryExecutor()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
