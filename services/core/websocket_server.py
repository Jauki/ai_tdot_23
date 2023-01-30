from __future__ import annotations

import base64
import io

import cv2
import imageio
import numpy as np
import websockets as ws
import logging
import json
import asyncio
from config import websocket_config as ws_config
from config import cascade_config
import abc
from typing import *
from service_manager import ServiceState
from service_results import FaceRecognitionResult, GlassesRecognitionResult, GlassesRecognitionResultType, \
    GenderEstimationResult

face_cascade_classifier = cv2.CascadeClassifier(cascade_config.FACE_CASCADE_CLASSIFIER_PATH)
face_recognition_service_state: ServiceState = ServiceState()
glasses_recognition_service_state: ServiceState = ServiceState()
gender_estimation_service_state: ServiceState = ServiceState()


class Channel(abc.ABC):
    def __init__(self, subclass_name: str):
        self._logger: logging.Logger = logging.getLogger(subclass_name)
        self._subscribers: set[ws.WebSocketServerProtocol] = set()
        self._message_handlers: dict[str, Callable[[ws.WebSocketServerProtocol, dict[str, any]], None]] = {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'publish': self._handle_publish,
        }

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def handle_message(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        message_type: str = message['type']

        if message_type in self._message_handlers:
            self._message_handlers[message_type](client, message)
        else:
            self._handle_unknown_message_type(message_type, client, message)

    def publish(self, payload: dict[str, any]):
        message: dict[str, any] = {
            'channel': self.name,
            'payload': payload,
        }
        ws.broadcast(self._subscribers, json.dumps(message))

    def _handle_unknown_message_type(self, message_type, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._logger.warning(f'unknown message type: {message_type}')

    def _handle_subscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._subscribers.add(client)

    def _handle_unsubscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        self._subscribers.remove(client)

    def _handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        message: dict[str, any] = {
            'channel': message['channel'],
            'payload': message['payload'],
        }
        # send to everyone except origin
        destinations = self._subscribers.difference({client})
        ws.broadcast(destinations, json.dumps(message))


class FaceRecognitionRequestChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['subscribe'] = self.__handle_subscribe

    @property
    def name(self) -> str:
        return 'face_recognition_request'

    def __handle_subscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        face_recognition_service_state.occupied = False

        super()._handle_subscribe(client, message)


class FaceRecognitionResultChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['publish'] = self.__handle_publish

    @property
    def name(self) -> str:
        return 'face_recognition_result'

    def __handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        result: FaceRecognitionResult = FaceRecognitionResult.from_dto(message['payload']['result'])

        face_recognition_service_state.result = result
        face_recognition_service_state.occupied = False

        super()._handle_publish(client, message)


face_recognition_request_channel: FaceRecognitionRequestChannel = FaceRecognitionRequestChannel()
face_recognition_result_channel: FaceRecognitionResultChannel = FaceRecognitionResultChannel()


class GlassesRecognitionRequestChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['subscribe'] = self.__handle_subscribe

    @property
    def name(self) -> str:
        return 'glasses_recognition_request'

    def __handle_subscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        glasses_recognition_service_state.occupied = False

        super()._handle_subscribe(client, message)


class GlassesRecognitionResultChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['publish'] = self.__handle_publish

    @property
    def name(self) -> str:
        return 'glasses_recognition_result'

    def __handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        result: GlassesRecognitionResult = GlassesRecognitionResult.from_dto(message['payload']['result'])

        glasses_recognition_service_state.result = result
        glasses_recognition_service_state.occupied = False

        super()._handle_publish(client, message)


glasses_recognition_request_channel: GlassesRecognitionRequestChannel = GlassesRecognitionRequestChannel()
glasses_recognition_result_channel: GlassesRecognitionResultChannel = GlassesRecognitionResultChannel()


class GenderEstimationRequestChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['subscribe'] = self.__handle_subscribe

    @property
    def name(self) -> str:
        return 'gender_estimation_request'

    def __handle_subscribe(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        gender_estimation_service_state.occupied = False

        super()._handle_subscribe(client, message)


class GenderEstimationResultChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['publish'] = self.__handle_publish

    @property
    def name(self) -> str:
        return 'gender_estimation_result'

    def __handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        result: GenderEstimationResult = GenderEstimationResult.from_dto(message['payload']['result'])

        gender_estimation_service_state.result = result
        gender_estimation_service_state.occupied = False

        super()._handle_publish(client, message)


gender_estimation_request_channel: GenderEstimationRequestChannel = GenderEstimationRequestChannel()
gender_estimation_result_channel: GenderEstimationResultChannel = GenderEstimationResultChannel()


class VideoStreamChannel(Channel):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._message_handlers['publish'] = self.__handle_publish

    @property
    def name(self) -> str:
        return 'video_stream'

    def __handle_publish(self, client: ws.WebSocketServerProtocol, message: dict[str, any]):
        # send to services

        payload: dict[str, any] = message['payload']
        frame: str = payload['frame']

        # convert base64 to image
        base64_frame: str = payload['frame']
        frame: np.ndarray = imageio.v3.imread(io.BytesIO(base64.b64decode(base64_frame)))
        frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # detect faces
        face_detections = face_cascade_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        if len(face_detections) > 0:
            # only process one face
            (x, y, w, h) = face_detections[0]

            # extract ROI (region of interest)
            roi: np.ndarray = np.copy(frame[y:y + h, x:x + w])

            # convert ROI to base64
            _, buffer = cv2.imencode('.png', roi)
            base64_roi: str = base64.b64encode(buffer).decode('utf-8')

            # send ROI to face recognition service
            if not face_recognition_service_state.occupied:
                face_recognition_service_state.occupied = True
                payload: dict[str, any] = {
                    'frame': base64_roi,
                }
                face_recognition_request_channel.publish(payload)  # send ROI to face recognition service

            if not glasses_recognition_service_state.occupied:
                glasses_recognition_service_state.occupied = True
                payload: dict[str, any] = {
                    'frame': base64_roi,
                }
                glasses_recognition_request_channel.publish(payload)

            if not gender_estimation_service_state.occupied:
                gender_estimation_service_state.occupied = True
                payload: dict[str, any] = {
                    'frame': base64_roi,
                }
                gender_estimation_request_channel.publish(payload)

            # highlight ROI
            color = (0, 255, 0)  # color in BGR
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

            # add landmarks to face
            result: GlassesRecognitionResult = glasses_recognition_service_state.result
            if result and result.result != GlassesRecognitionResultType.NO_FACE:
                for i in range(68):
                    landmark_x = result.landmarks[i][0] + x
                    landmark_y = result.landmarks[i][1] + y
                    cv2.circle(frame, (landmark_x, landmark_y), radius=2, color=(0, 0, 255), thickness=-1)
                    # cv2.putText(frame, str(i), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
                    #             cv2.LINE_AA)

            # convert frame to base64
            _, buffer = cv2.imencode('.png', frame)
            base64_frame: str = base64.b64encode(buffer).decode('utf-8')

            # set frame with highlighted ROI
            message['payload'] = {
                'frame': base64_frame,
            }

        super()._handle_publish(client, message)


video_stream_channel: VideoStreamChannel = VideoStreamChannel()


class WebSocketServer:
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__connections: set[ws.WebSocketServerProtocol] = set()
        self.__channels: dict[str, Channel] = {
            video_stream_channel.name: video_stream_channel,
            face_recognition_request_channel.name: face_recognition_request_channel,
            face_recognition_result_channel.name: face_recognition_result_channel,
            glasses_recognition_request_channel.name: glasses_recognition_request_channel,
            glasses_recognition_result_channel.name: glasses_recognition_result_channel,
            gender_estimation_request_channel.name: gender_estimation_request_channel,
            gender_estimation_result_channel.name: gender_estimation_result_channel,
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
