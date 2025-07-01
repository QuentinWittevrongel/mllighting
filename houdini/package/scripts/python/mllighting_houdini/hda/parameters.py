import hdefereval
import hou

from mllighting import log

from mllighting_houdini import server


logger = log.LoggerManager.get_logger(__name__)


def start_server(node: hou.OpNode):
    """Executed when the start server button is pressed.

    Args:
        node: The node to start the server from.
    """
    # Asynio can only be run in the main thread.
    # Parameter callbacks run in a different thread, we defer the execution
    # for it to be executed in the main thread.
    hdefereval.executeDeferred(server.start_server, node)


def stop_server(node: hou.OpNode):
    """Executed when the stop server button is pressed.

    Args:
        node: The node to stop the server from.
    """
    # Execute the stop server in the main thread.
    hdefereval.executeDeferred(server.stop_server, node)


def clear_lights(node: hou.OpNode):
    """Executed when the clear lights button is pressed.

    Args:
        node: The node to clear the lights from.
    """
    inlineusd_node = node.node('IN_RESULTS')
    inlineusd_node.parm('usdsource').set('')


def render_to_drawing(node: hou.OpNode):
    """Render the scene and send the albedo to the drawing application.

    Args:
        node: The node to execute the render from.
    """
    server_manager = server.get_server_manager(node)
    if server_manager is None:
        logger.warning(f'No server manager defined for {node.path()}')
        return

    server_manager.loop.create_task(server.render_to_drawing(node))
