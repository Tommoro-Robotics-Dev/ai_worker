#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.executors import MultiThreadedExecutor
import sounddevice as sd
import numpy as np
import threading
import tempfile
import requests
import os
import time
import soundfile as sf

from tr_modules_interfaces.action import RecordSpeech, LLMAction

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self._llm_client = ActionClient(self, LLMAction, 'llm_action')
        self._action_server = ActionServer(
            self,
            RecordSpeech,
            '/stt/record',
            self.execute_callback,
            goal_callback=lambda _: rclpy.action.GoalResponse.ACCEPT,
            cancel_callback=lambda _: rclpy.action.CancelResponse.ACCEPT
        )

        self.device_id = 25 # Adjust this to your microphone device ID
        self.samplerate = self.get_device_samplerate(self.device_id)
        self.get_logger().info("STT Action Server ready")

    def get_device_samplerate(self, device_id):
        device_info = sd.query_devices(device_id, "input")
        return int(device_info["default_samplerate"])

    def execute_callback(self, goal_handle):
        self.get_logger().info("Recording started...")
        audio_data = []
        stop_flag = False

        def audio_callback(indata, frames, time_info, status):
            audio_data.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype='int16',
            callback=audio_callback,
            device=self.device_id
        )

        try:
            stream.start()
            while rclpy.ok():
                time.sleep(0.1)
                if goal_handle.is_cancel_requested:
                    stop_flag = True
                    break
        finally:
            stream.stop()
            stream.close()

        self.get_logger().info("Processing audio...")
        result = RecordSpeech.Result()

        try:
            if not audio_data:
                raise ValueError("No audio data recorded")

            audio_np = np.concatenate(audio_data, axis=0).astype(np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_np, self.samplerate)
                temp_audio_path = f.name

            with open(temp_audio_path, "rb") as audio_file:
                headers = {
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}"
                }
                start_time = time.time()
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files={
                        "file": (os.path.basename(temp_audio_path), audio_file, "audio/wav")
                    },
                    data={"model": "whisper-1"}
                )
                inference_duration = time.time() - start_time
                self.get_logger().info(f"Inference time: {inference_duration:.2f} seconds")
            os.remove(temp_audio_path)

            if response.status_code == 200:
                text = response.json().get("text", "").strip()
                if text:
                    result.success = True
                    result.result_text = text
                    result.message = "Success"
                    goal_handle.succeed()
                else:
                    raise ValueError("No speech detected")
            else:
                raise ValueError(f"API Error: {response.status_code} {response.text}")

        except Exception as e:
            result.success = False
            result.result_text = ""
            result.message = str(e)
            goal_handle.abort()

        return result


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()