from __future__ import annotations

import base64
import io
import logging.config
import asyncio
import time

import cv2
import imageio
import numpy as np

from glasses_recognition_service import GlassesRecognitionResult, GlassesRecognitionService
from config import websocket_config as ws_config
import websocket_client as ws_client
import os
import websockets as ws

LOGGING_CONFIG_PATH: str = os.path.join(os.getcwd(), 'config', 'logging.conf')

client = ws_client.WebSocketClient(
    host='localhost',
    port=5678,
)

glasses_recognition_service = GlassesRecognitionService()


async def handle_message(server: ws.WebSocketClientProtocol, channel: str, payload: dict[str, any]):
    # convert base64 to image
    base64_frame: str = payload['frame']
    frame: np.ndarray = imageio.v3.imread(io.BytesIO(base64.b64decode(base64_frame)))
    frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # predict
    result: GlassesRecognitionResult = glasses_recognition_service.detect_glasses(frame)

    # send back result
    payload: dict[str, any] = {
        'result': result.to_dto(),
    }
    await client.publish(payload, 'glasses_recognition_result')


async def main():
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    handle_messages = await client.connect()
    await client.subscribe('glasses_recognition_request')
    client.on_message = handle_message
    await asyncio.gather(
        handle_messages,
    )


if __name__ == '__main__':
    asyncio.run(main())
