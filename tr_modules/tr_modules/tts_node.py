#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from tr_modules_interfaces.action import TextToSpeech
import os
import requests
import sounddevice as sd
import numpy as np
import io
import wave
from pydub import AudioSegment
import scipy.signal
import time

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.get_logger().info("Starting TTSNode with OpenAI TTS (model: tts-1)...")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.get_logger().error("OPENAI_API_KEY not set!")
            return

        self.action_server = ActionServer(
            self,
            TextToSpeech,
            '/tts/speak',
            self.execute_callback
        )

        sd.default.device = 'hw:3,0'  # Jabra or relevant device
        sd.default.samplerate = 48000  # OpenAI TTS returns audio at 24 kHz

    async def execute_callback(self, goal_handle):
        text = goal_handle.request.text
        self.get_logger().info(f"[TTS] Received text: {text}")

        try:
            start_time = time.time()
            audio_data = self.call_openai_tts(text)
            end_time = time.time()  
            inference_time = end_time - start_time
            self.get_logger().info(f"[TTS] Inference time: {inference_time:.3f} seconds")

            self.play_audio(audio_data)
            goal_handle.succeed()
            return TextToSpeech.Result(success=True, message="Playback successful.")
        except Exception as e:
            self.get_logger().error(f"[TTS] Error: {str(e)}")
            goal_handle.abort()
            return TextToSpeech.Result(success=False, message=str(e))

    def call_openai_tts(self, text):
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "voice": "nova",
            "input": text
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.text}")

        return response.content 

    def play_audio(self, audio_bytes):
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(audio.array_type).max  

        if audio.frame_rate != 48000:
            samples = scipy.signal.resample_poly(samples, 48000, audio.frame_rate)
        try:
            sd.play(samples, samplerate=48000)
            sd.wait()
        except Exception as e:
            raise RuntimeError(f"Error playing audio: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()