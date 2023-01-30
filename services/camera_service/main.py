from __future__ import annotations
import asyncio
import cv2
import base64
import os
import time
import logging.config
import websockets as ws
import websocket_client as ws_client
from config import websocket_config as ws_config, video_stream_config

LOGGING_CONFIG_PATH: str = os.path.join(os.getcwd(), 'config', 'logging.conf')

client = ws_client.WebSocketClient(
    host='localhost',
    port=5678,
)


async def stream_images():
    camera = cv2.VideoCapture(0)

    while True:
        # get frame
        _, frame = camera.read()

        # scale down
        dims = frame.shape
        width, height = int(dims[1] * video_stream_config.SCALE), int(dims[0] * video_stream_config.SCALE)
        new_dims = (width, height)
        downscaled_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)

        # encode to base-64
        _, buffer = cv2.imencode('.png', downscaled_frame)
        base64_image: str = base64.b64encode(buffer).decode('utf-8')

        # send image
        payload: dict[str, any] = {
            'frame': base64_image,
        }
        await client.publish(payload, 'video_stream')

        # sleep
        await asyncio.sleep(0.01)


async def main():
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    handle_messages = await client.connect()
    await asyncio.gather(
        handle_messages,
        stream_images(),
    )


if __name__ == '__main__':
    asyncio.run(main())
