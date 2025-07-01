import abc
import asyncio
import json
import queue
import typing

from mllighting import log


logger = log.LoggerManager.get_logger(__name__)


class Server:
    """Handle communication."""

    def __init__(self, address: str, port: int, command_queue: queue.Queue):
        self._server = None

        self._address = address
        self._port = port
        self._command_queue = command_queue

        self._shutdown_event = asyncio.Event()

    async def start_server(self):
        """Start the server."""
        logger.debug(f'Starting server {self._address}:{self._port}')

        # Reset the shutdown event.
        self._shutdown_event.clear()

        # Create the asyncio server.
        self._server = await asyncio.start_server(
                self._handle_client,
                self._address,
                self._port)

        try:
            # Wait for the server to finish or the shutdown event to be set.
            server_task = asyncio.create_task(self._server.serve_forever())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED)

            # Cancel pending tasks
            for task in pending:
                task.cancel()

        except Exception as e:
            logger.error(f'Server error: {e}')
        finally:
            await self._stop_server()

    async def stop_server(self):
        """Stop the server."""
        logger.debug('Stop command received')
        self._shutdown_event.set()

    async def _stop_server(self):
        """Internal server shutdown method."""
        logger.debug(f'Stopping server {self._address}:{self._port}')

        try:
            self._server.close()
        except AttributeError:
            # TODO: Issue with Houdini:
            # AttributeError: 'AsyncioAcceptor' object has no attribute
            # 'detach'
            pass
        await self._server.wait_closed()
        logger.debug('Server stopped')

    async def _handle_client(
            self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Server callback.

        Args:
            reader: The reader data stream.
            writer: The writer data stream.
        """
        logger.debug('Received data')
        try:
            # Read data from the reader.
            data = await reader.read(8192)
            if not data:
                logger.debug('No data')
                return

            # Decode the command.
            command = json.loads(data.decode())
            cmd = command.get('command')
            arguments = command.get('arguments', {})

            # Put the received command into the queue to be executed by the
            # main loop.
            logger.debug(f'Received command {cmd} with arguments {arguments}')
            try:
                self._command_queue.put(
                    (cmd, arguments),
                    timeout=10
                )
                response = {'message': 'Got it'}
            except Exception as queue_execption:
                response = {
                    'error':
                    f'Error while putting command to queue: {queue_execption}'}

            # Send a response to the client.
            # The actual command may be executed later in the main tread.
            logger.debug(f'Sending response {response}')
            writer.write(json.dumps(response).encode())
            await writer.drain()

        except Exception as e:
            logger.error(f'Server callback error: {e}')
        finally:
            writer.close()
            await writer.wait_closed()


class ServerManager(abc.ABC):
    """Handle the server in the background."""

    def __init__(self):
        # The server is made to run in background to not block the main
        # application. But applications need to execute the commands in their
        # main thread.
        # For this we create a queue object that will be filled by the
        # background thread, and we will periodically check the queue to
        # execute the commands.
        # The queue pooling system is dependent on the host application.
        self._command_queue = queue.Queue()

        # The list of registered command to execute.
        self._commands: dict[str, typing.Callable] = {}

    @property
    def command_queue(self) -> queue.Queue:
        """The command queue."""
        return self._command_queue

    @abc.abstractmethod
    def start_server(self, address: str, port: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def stop_server(self):
        raise NotImplementedError()

    def register_command(self, command: str, callback: typing.Callable):
        """Register a new command.

        Commands with the same command name will be replaced.

        Args:
            command: The command to register.
            callback: The function to execute.
        """
        self._commands[command] = callback

    def process_command_queue(self):
        """Read the command queue to execute commands in main thread.

        This method is part of the command queue pooling system that must be
        reimplemented in the host application to be called periodically.
        """
        while not self._command_queue.empty():
            # Get from the queue.
            try:
                command, kwargs = self._command_queue.get(timeout=10)
            except queue.Empty as e:
                logger.error(f'Command queue get method timeout reached: {e}')
                continue
            except Exception as e:
                logger.error(f'Error while getting from command queue: {e}')
                continue

            logger.debug(
                f'Received command {command} with arguments {kwargs} '
                'in main thread')

            # Get the function to execute.
            func = self._commands.get(command, None)
            if func is None:
                logger.error(f'No function assigned to command {command}')
                continue

            try:
                self.process_command(func, kwargs)
            except Exception as e:
                logger.error(
                    f'Exception while executing {func} with args {kwargs} '
                    f'in main thread: {e}')

    def process_command(self, function: typing.Callable, kwargs: dict):
        """Process the command read from the command queue.

        Args:
            function: The function to execute.
            kwargs: Keyword arguments to pass to the function.
        """
        function(**kwargs)
