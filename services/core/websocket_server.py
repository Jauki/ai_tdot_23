from __future__ import annotations
import websockets as ws
import logging
import json
import asyncio
from config import websocket_config as ws_config
import abc
from typing import *


class Channel(abc.ABC):
    def __init__(self, subclass_name: str):
        self._logger: logging.Logger = logging.getLogger(subclass_name)
        self._subscribers: set[ws.WebSocketServerProtocol] = set()
        self._message_handlers: dict[str, Callable[[ws.WebSocketServerProtocol, dict[str, any]], None]] = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'publish': self._handle_publish,
        }

    def handle_message(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        message_type: str = message['type']

        if message_type in self._message_handlers:
            self._message_handlers[message_type](client, message)
        else:
            self._handle_unknown_message_type(message_type, client, message)

    def _handle_unknown_message_type(self, message_type, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._logger.warning(f'unknown message type: {message_type}')

    def _handle_subscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._subscribers.add(client)

    def _handle_unsubscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._subscribers.remove(client)

    def _handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        ws.broadcast(self._subscribers, json.dumps(message))


class VideoStreamChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        # extend handlers
        self._message_handlers['publish_frame'] = self._handle_publish

    def _handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._logger.debug('received video stream publish message')
        super()._handle_publish(client, message)


class WebSocketServer:
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__connections: set[ws.WebSocketServerProtocol] = set()
        self.__channels: dict[str, Channel] = {
            'video_stream': VideoStreamChannel()
        }

    async def __handle_client(self, client: ws.WebSocketServerProtocol):
        async for message in client:
            message = json.loads(message)
            channel_name: str = message['channel']

            if channel_name in self.__channels:
                self.__logger.debug(f'received message on channel={channel_name}')
                channel = self.__channels[channel_name]
                channel.handle_message(client, message)
            else:
                self.__logger.warning(f'unknown channel={channel_name}; discarding message!')

    async def __handle_connect(self, client: ws.WebSocketServerProtocol):
        self.__connections.add(client)

        self.__logger.info(f'new client connected')
        self.__logger.info(f'active connection count: {len(self.__connections)}')
        try:
            await self.__handle_client(client)
        finally:
            self.__connections.remove(client)

            self.__logger.info(f'client disconnected')
            self.__logger.info(f'active connection count: {len(self.__connections)}')

    async def run(self):
        async with ws.serve(self.__handle_connect, ws_config.HOST, ws_config.PORT):
            self.__logger.info('server started successfully')
            await asyncio.Future()  # wait forever
