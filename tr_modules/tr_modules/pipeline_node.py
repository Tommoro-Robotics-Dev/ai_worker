#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
import threading
import sys
import select
from datetime import datetime
from tr_modules_interfaces.action import RecordSpeech, LLMAction, TextToSpeech

class PipelineNode(Node):
    def __init__(self):
        super().__init__('pipeline_node')
        self.recording_client = ActionClient(self, RecordSpeech, '/stt/record')
        self.llm_client       = ActionClient(self, LLMAction,    'llm_action')
        self.tts_client       = ActionClient(self, TextToSpeech, '/tts/speak')
        self.shutdown_flag    = threading.Event()
        self.is_recording     = False
        self._stt_goal_handle = None

        self._time_stt_end    = None
        self._time_llm_resp   = None
        self._time_tts_send   = None

        self.input_thread = threading.Thread(target=self.input_handler)
        self.input_thread.daemon = True
        self.input_thread.start()

        self.get_logger().info("Pipeline control ready (ENTER=record, X=exit)")

    def timestamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def input_handler(self):
        print("\n>> Press ENTER to start recording, X to exit")
        while not self.shutdown_flag.is_set():
            rlist, _, _ = select.select([sys.stdin], [], [], 1)
            if rlist:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'x':
                    self.shutdown_flag.set()
                    self.get_logger().info(f"[{self.timestamp()}] Exit requested")
                    return
                else:
                    self.toggle_recording()
                    print("\n>> Press ENTER to stop recording, X to exit")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.get_logger().info(f"[{self.timestamp()}] Starting recording...")
        self.is_recording = True

        if not self.recording_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"[{self.timestamp()}] STT server not available")
            self.is_recording = False
            return

        goal_future = self.recording_client.send_goal_async(RecordSpeech.Goal())
        goal_future.add_done_callback(self.recording_goal_callback)

    def stop_recording(self):
        if not self.is_recording:
            return

        self._time_stt_end = self.timestamp()
        self.get_logger().info(f"[{self._time_stt_end}] Stopping recording...")
        self.is_recording = False

        if self._stt_goal_handle is not None:
            self._stt_goal_handle.cancel_goal_async()
            self._stt_goal_handle = None

    def recording_goal_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"[{self.timestamp()}] Recording rejected")
            return

        self._stt_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recording_result_callback)

    def recording_result_callback(self, future):
        response = future.result().result
        if not response.success:
            self.get_logger().error(f"[{self.timestamp()}] STT failed: {response.message}")
            self.is_recording = False
            return

        text = response.result_text.strip()
        self.get_logger().info(f"[{self.timestamp()}] STT result: \"{text}\"")
        self.process_llm(text)
        self.is_recording = False

    def process_llm(self, prompt):
        if not self.llm_client.wait_for_server(timeout_sec=5.0):
            #self.get_logger().error(f"[{self.timestamp()}] LLM server not available")
            return

        goal_msg = LLMAction.Goal()
        goal_msg.prompt = prompt

        send_goal_future = self.llm_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.llm_goal_callback)

    def llm_goal_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            #self.get_logger().warn(f"[{self.timestamp()}] LLM goal rejected")
            return

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.llm_result_callback)

    def llm_result_callback(self, future):
        get_result_response = future.result()
        reply_text = get_result_response.result.reply

        self._time_llm_resp = self.timestamp()
        #self.get_logger().info(f"[{self._time_llm_resp}] LLM response: \"{reply_text}\"")


def main(args=None):
    rclpy.init(args=args)
    node = PipelineNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_flag.set()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()