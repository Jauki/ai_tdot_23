import os
import websocket_server as ws_server
import logging.config
import asyncio

LOGGING_CONFIG_PATH: str = os.path.join(os.getcwd(), 'config', 'logging.conf')


def main():
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    server = ws_server.WebSocketServer()
    asyncio.run(server.run())


if __name__ == '__main__':
    main()
