from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS


class SocketBackendMeta(type):
    """
    Metaclass for Singleton
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SocketBackend(metaclass=SocketBackendMeta):
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'secret!'
        CORS(self.app, resources={r'/*': {'origins': '*'}})
        self.socket = SocketIO(self.app, cors_allowed_origins='*')
        self.socket.on('register', self.handle_register_user)
        self.socket.on('save_image', self.handle_save_image)

    def handle_register_user(self):
        """route to register a user with a specified name"""
        # TODO: make user registration
        data = request.json
        print(jsonify(data))
        return "";

    def handle_save_image(self, data):
        """Websocket event to retrieve image-data from the client"""
        # TODO: make image save
        print(data)
        # if successful:
        emit('success')

    def run(self):
        self.socket.run(self.app, debug=True, port=1234)
