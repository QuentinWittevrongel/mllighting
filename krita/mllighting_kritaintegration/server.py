import asyncio
import queue
import threading
import typing

import krita

from PyQt5 import QtCore

from mllighting import log
from mllighting.communication import server


logger = log.LoggerManager.get_logger(__name__)


class KritaServerManager(server.ServerManager):

    def __init__(self):
        super().__init__()

        # The server loop.
        self._loop = None
        self._loopthread = None
        self._server = None

        # The queue pooling system.
        self._checkloop = QtCore.QTimer()
        self._checkloop.timeout.connect(self.process_command_queue)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def start_server(self, address: str, port: int):
        # Start the server in a background thread.
        logger.debug(f'Creating start server {address}:{port} task')
        self._loopthread = threading.Thread(
            target=self._event_loop,
            args=(address, port, self._command_queue),
            daemon=True)
        self._loopthread.start()

        # Start the command processing loop.
        logger.debug('Start the command processing loop')
        self._checkloop.start(1000)

    def stop_server(self):
        if self._server is None:
            logger.warning('No running server')
            return

        logger.debug('Stopping server')
        future = asyncio.run_coroutine_threadsafe(
            self._server.stop_server(), self._loop)
        future.result()

        # Stop the command processing loop.
        logger.debug('Stop the command processing loop')
        self._checkloop.stop()

    def process_command(self, function: typing.Callable, kwargs: dict):
        krita_instance = krita.Krita.instance()
        function(krita_instance, **kwargs)

    def _event_loop(self, address: str, port: int, command_queue: queue.Queue):
        """The event loop executed in a background thread.

        Args:
            address: The server address.
            port: The server port.
            command_queue: The command queue.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._server = server.Server(address, port, command_queue)

        try:
            self._loop.run_until_complete(self._server.start_server())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f'Event loop error: {e}')
        finally:
            self._loop.close()
