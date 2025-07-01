import hou

from mllighting_houdini import server


def on_deleted(kwargs: dict):
    """Executed when the node is deleted.

    Args:
        kwargs: The event dictionary.
    """
    node: hou.OpNode = kwargs['node']
    server.stop_server(node)
