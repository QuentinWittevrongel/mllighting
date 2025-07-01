import asyncio
import json
import os

import krita

from mllighting import log

from mllighting_kritaintegration import server


ALBEDO_LAYER_NAME = 'albedo'
logger = log.LoggerManager.get_logger(__name__)


def get_albedo_layer(document: krita.Document) -> krita.FileLayer | None:
    """Get the albedo layer in the given document.

    Returns:
        The albedo layer, None if not found.
    """
    return document.nodeByName(ALBEDO_LAYER_NAME)


def initialize_albedo_layer(krita_instance: krita.Krita) -> krita.FileLayer:
    """Create the albedo layer.

    Args:
        krita_instance: The Krita instance.

    Returns:
        The albedo layer.
    """
    # Get the current document.
    document = krita_instance.activeDocument()

    # Check if the layer already exists.
    albedo_layer = get_albedo_layer(document)
    if albedo_layer is not None:
        # Delete it.
        albedo_layer.remove()

    # Create the albedo layer.
    albedo_layer = document.createFileLayer(
        ALBEDO_LAYER_NAME, '', 'None')

    # Add the node in the document, at the bottom..
    root_node = document.rootNode()
    child_nodes = root_node.childNodes()
    above_node = child_nodes[0] if child_nodes else None
    root_node.addChildNode(albedo_layer, above_node)
    albedo_layer.setLocked(True)

    return albedo_layer


def albedo_received(krita_instance: krita.Krita, path: str):
    """Function executed when an albedo is received.

    Args:
        krita_instance: The Krita instance.
        path: The albedo image file path.
    """
    document = krita_instance.activeDocument()

    albedo_layer = get_albedo_layer(document)
    if albedo_layer is None:
        albedo_layer = initialize_albedo_layer(krita_instance)

    albedo_layer.setLocked(False)
    albedo_layer.setProperties(
        path,
        albedo_layer.scalingMethod(),
        albedo_layer.scalingFilter())
    albedo_layer.setLocked(True)


def send_beauty(
        document: krita.Document,
        address: str,
        port: int,
        server_manager: server.KritaServerManager):
    """Send the beauty from the document.

    Args:
        document: The document to export from.
        address: The address to send the beauty to.
        port: The port to send the beauty to.
        server_manager: The server manager to send with.
    """
    # Export the beauty in the same directory than the albedo layer.
    albedo_layer = get_albedo_layer(document)
    if albedo_layer is None:
        logger.warning('No albedo layer')
        return

    albedo_path = albedo_layer.path()
    if albedo_path is None:
        logger.warning('Albedo layer does not contain any path')

    render_directory = os.path.dirname(albedo_path)

    beauty_file_path = os.path.join(render_directory, 'beauty.png')
    is_exported = document.exportImage(beauty_file_path, krita.InfoObject())
    if not is_exported:
        raise Exception(f'Could not export beauty to {beauty_file_path}')

    # Execute the
    future = asyncio.run_coroutine_threadsafe(
        _send_beauty(beauty_file_path, address, port),
        server_manager.loop)
    try:
        future.result(timeout=10)
    except asyncio.TimeoutError as e:
        logger.error(f'Send beauty timeout: {e}')
        return


async def _send_beauty(
        beauty_file_path: str,
        address: str,
        port: int):
    """Send the beauty through asyncio.

    Args:
        beauty_file_path: The beauty file to send.
        address: The address to send to.
        port: The port to send to.
    """
    reader, writer = await asyncio.open_connection(address, port)
    command = {
        'command': 'send_beauty',
        'arguments': {
            'path': beauty_file_path}}

    writer.write(json.dumps(command).encode())
    await writer.drain()

    writer.close()
    await writer.wait_closed()
