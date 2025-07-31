#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.action import GoalResponse, CancelResponse
from tr_modules_interfaces.action import LLMAction, TextToSpeech
#from rby1_interfaces.srv import LLMFeedbackToGUI
from openai import OpenAI
import os
import numpy as np
import json
from scipy.spatial.distance import cosine
import time

FIFO_PATH = '/tmp/ai_worker_cmd'
if not os.path.exists(FIFO_PATH):
    os.mkfifo(FIFO_PATH)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMNode(Node):

    def __init__(self):
        super().__init__('llm_node')

        self._action_server = ActionServer(
            self,
            LLMAction,
            'llm_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self._tts_client = ActionClient(self, TextToSpeech, '/tts/speak')
        #self._gui_client = self.create_client(LLMFeedbackToGUI, '/gui_action')

        self.get_logger().info("LLMNode is up and running")

        # Hardcoded predefined prompts and embeddings
        self.prompts = [
            "please go to the initial position",
            "por favor ve a la posición inicial",
            "처음 위치로 이동해주세요",
            "please be prepared to work on the desk",
            "por favor prepárate para trabajar en el escritorio",
            "책상에서 작업할 준비를 해주세요",
        ]
        self.embeddings = []
        self.threshold = 0.45
        for prompt in self.prompts:
            emb = client.embeddings.create(model="text-embedding-3-small", input=prompt).data[0].embedding
            self.embeddings.append(emb)
        self.get_logger().info("Hardcoded prompts embedded and ready.")

    def goal_callback(self, goal_request):
        self.get_logger().info('Received LLM goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received LLM cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        prompt = goal_handle.request.prompt
        self.get_logger().info(f"[LLM] Received prompt: {prompt}")

        try:
            start_time = time.time()
            emb = client.embeddings.create(model="text-embedding-3-small", input=prompt).data[0].embedding
            max_sim = 0.0
            best_match = None

            for p, e in zip(self.prompts, self.embeddings):
                sim = 1 - cosine(emb, e)
                if sim > max_sim:
                    max_sim = sim
                    best_match = p

            if max_sim > self.threshold:
                self.get_logger().info(f"[LLM] Action detected: {best_match}")

                # Mapeo de frases a comandos FIFO
                mapping = {
                    "please go to the initial position":                      "pose 2",
                    "por favor ve a la posición inicial":                  "pose 2",
                    "처음 위치로 이동해주세요":                               "pose 2",
                    "please be prepared to work on the desk":           "pose 1",
                    "por favor prepárate para trabajar en el escritorio":"pose 1",
                    "책상에서 작업할 준비를 해주세요":                         "pose 1",
                }

                cmd_str = mapping.get(best_match.lower())
                if cmd_str:
                    # Envía el comando al poser vía FIFO
                    os.system(f"echo '{cmd_str}' > {FIFO_PATH}")

                    # Confirma la acción por TTS
                    tts_text = f"Executing {cmd_str}"
                    await self.call_tts_action(tts_text)

                goal_handle.succeed()
                return LLMAction.Result(reply=best_match)
            else:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are RB-Y1, a humanoid robot developed by Rainbow Robotics. Your job is to understand natural language commands from users and help them through actions or information. Respond clearly, briefly, and always stay in character as the robot RB-Y1. Do not speculate or make up facts. If a user gives a command that resembles a robotic task (e.g. placing, grabbing, pouring), summarize it precisely. Otherwise, answer like a robot would, politely and efficiently."},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
                self.get_logger().info("[LLM] No action was detected. Sending response string to TTS Node.")
                duration = time.time() - start_time
                self.get_logger().info(f"[LLM] Inference time: {duration:.3f} seconds | {duration / max(1, len(prompt)):.5f} sec/char")
                await self.call_tts_action(reply)
                goal_handle.succeed()
                return LLMAction.Result(reply=reply)

            goal_handle.succeed()
            return LLMAction.Result(reply=best_match)

        except Exception as e:
            self.get_logger().error(f"[LLM] Error: {str(e)}")
            goal_handle.abort()
            return LLMAction.Result(reply=f"Error: {str(e)}")

    async def call_tts_action(self, text):
        if not self._tts_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('[LLM] TTS action server not available.')
            return

        goal_msg = TextToSpeech.Goal()
        goal_msg.text = text

        self.get_logger().info(f"[LLM] Sending goal to TTS: {text}")
        send_goal_future = self._tts_client.send_goal_async(goal_msg)

        goal_handle = await send_goal_future
        if not goal_handle.accepted:
            self.get_logger().warn("[LLM] TTS goal rejected.")
            return

        self.get_logger().info("[LLM] TTS goal accepted.")
        result_future = goal_handle.get_result_async()
        result = await result_future

        if result.result.success:
            self.get_logger().info("[LLM] TTS playback successful.")
        else:
            self.get_logger().error(f"[LLM] TTS playback failed: {result.result.message}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()