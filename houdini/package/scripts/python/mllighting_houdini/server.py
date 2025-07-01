import asyncio
import json
import os
import typing

import hou

from hutil.PySide import QtCore

from mllighting import log
from mllighting.communication import server

from mllighting_houdini import commands


logger = log.LoggerManager.get_logger(__name__)


class HoudiniServerManager(server.ServerManager):

    def __init__(self, node: hou.OpNode):
        super().__init__()
        self._node = node

        # The server loop.
        self._loop = None

        # The queue pooling system.
        self._checkloop = QtCore.QTimer(parent=hou.ui.mainQtWindow())
        self._checkloop.timeout.connect(self.process_command_queue)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def node(self) -> hou.OpNode:
        """The node associated with the server manager."""
        return self._node

    def start_server(self, address: str, port: int):
        # Since Houdini 20 and their haio module, we can not run asyncio loops
        # in background thread and call loop.run_forever().
        logger.debug(f'Creating start server {address}:{port} task')
        self._loop = asyncio.new_event_loop()

        asyncio.set_event_loop(self._loop)
        self._server = server.Server(address, port, self.command_queue)
        self._loop.create_task(self._server.start_server())

        # Start the command processing loop.
        logger.debug('Start the command processing loop')
        self._checkloop.start(1000)

    def stop_server(self):
        logger.debug('Stopping server')
        self._loop.create_task(self._server.stop_server())

        # Stop the command processing loop.
        logger.debug('Stop the command processing loop')
        self._checkloop.stop()

    def process_command(self, function: typing.Callable, kwargs: dict):
        function(self._node, **kwargs)


def start_server(node: hou.OpNode):
    """Internal start server function executed in the main thread.

    Args:
        node: The node to start the server from.
    """
    try:
        # Get the server manager from the node data.
        server_manager = get_server_manager(node)
        if server_manager is None:
            # Create a new server manager for the node.
            server_manager = HoudiniServerManager(node)
            # Register the commands.
            server_manager.register_command(
                'send_beauty', commands.beauty_received)
            # Store the server in the node data to not let the garbage
            # collector delete it.
            set_server_manager(node, server_manager)

        # Get the server info from the node parameters.
        address = node.parm('serveraddress').evalAsString()
        port = node.parm('serverport').evalAsInt()
        logger.debug(
            f'Starting server {address}:{port} from node {node.path()}')
        server_manager.start_server(address, port)

    except hou.ObjectWasDeleted:
        logger.error('Node was deleted')


def stop_server(node: hou.OpNode):
    """Internal stop server function executed in the main thread.

    Args:
        node: The node to stop the server from.
    """
    try:
        # Get the server manager from the node cache.
        server_manager = get_server_manager(node)
        if server_manager is None:
            logger.debug(f'The node {node.path()} has no server manager')
            return
        logger.debug(f'Stopping server from node {node.path()}')
        server_manager.stop_server()
    except hou.ObjectWasDeleted:
        logger.error('Node was deleted')


def get_server_manager(node: hou.OpNode) -> HoudiniServerManager | None:
    """Get the server manager from the node.

    Args:
        node: The node to get the server manager from.

    Returns:
        The server manager. None if not found.
    """
    return node.cachedUserData('mllighting_server')


def set_server_manager(node: hou.OpNode, server_manager: HoudiniServerManager):
    """Set the server manager for the node.

    Args:
        node: The node to set the server manager to.
        server_manager: The server manager.
    """
    node.setCachedUserData('mllighting_server', server_manager)


async def render_to_drawing(node: hou.OpNode):
    """Render and send the result to the drawing application.

    Args:
        node: The ndoe to render from.
    """
    # Get the render rop to execute.
    render_node: hou.RopNode = node.node('OUT_RENDER')
    if render_node is None:
        raise Exception('No rop node OUT_RENDER found')

    # Execute the render.
    render_node.render()

    # Get the albedo file.
    render_directory = node.parm('renderdirectory').evalAsString()
    albedo_file_path = os.path.join(render_directory, 'albedo.png')
    if not os.path.exists(albedo_file_path):
        raise FileNotFoundError(f'No albedo file {albedo_file_path}')

    # Send the albedo to the drawing application.
    address = node.parm('drawappaddress').evalAsString()
    port = node.parm('drawappport').evalAsInt()
    logger.debug(f'Sending the albedo to {address}:{port}')
    reader, writer = await asyncio.open_connection(address, port)
    command = {
        'command': 'send_albedo',
        'arguments': {
            'path': albedo_file_path}}
    writer.write(json.dumps(command).encode())
    await writer.drain()

    # Wait for an answer.
    await reader.read(8192)

    writer.close()
    await writer.wait_closed()
