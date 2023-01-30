from __future__ import annotations
import websockets as ws
import abc
import logging
import asyncio
import json
from typing import *


class WebSocketClient:
    def __init__(self, host: str, port: int):
        self.__url = f'ws://{host}:{port}'
        self.__logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.__server: ws.WebSocketClientProtocol | None = None
        self.on_message: Callable[[ws.WebSocketClientProtocol, str, dict[str, any]], None] | None = None

    async def connect(self) -> Coroutine:
        self.__server: ws.WebSocketClientProtocol = await ws.connect(self.__url)
        self.__logger.debug('successfully established a connection')
        return self.__handle_messages()

    async def __handle_messages(self):
        while True:
            message = await self.__server.recv()
            message = json.loads(message)
            channel: str = message['channel']
            payload: dict[str, any] = message['payload']

            if self.on_message:
                self.on_message(self.__server, channel, payload)

    async def subscribe(self, channel: str):
        message: dict[str, any] = {
            'channel': channel,
            'type': 'subscribe',
        }
        await self.__server.send(json.dumps(message))

    async def publish(self, payload: dict[str, any], channel: str):
        message: dict[str, any] = {
            'channel': channel,
            'type': 'publish',
            'payload': payload,
        }
        await self.__server.send(json.dumps(message))
