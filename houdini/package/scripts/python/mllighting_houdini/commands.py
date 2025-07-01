import os

import hou

from pxr import Gf, Sdf

import torch

from mllighting import log
from mllighting.ml import inference, network


logger = log.LoggerManager.get_logger(__name__)


def beauty_received(node: hou.OpNode, path: str):
    logger.debug(f'Received the beauty {path}')

    render_directory = os.path.dirname(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f'Using device {device}')

    # Load the model.
    checkpoint = node.parm('checkpoint').evalAsString()
    model = network.load_model(
        checkpoint=checkpoint,
        device=device)

    # Predict the values.
    predicted_lights = inference.run_inference(
        model, render_directory, device=device)

    # Format the infered values.
    # We only predict 3 values, but this can change if we predict more lights
    # or attributes.
    stride = 3
    light_data = []
    for i in range(0, len(predicted_lights), stride):
        light_data.append({
            'matrix': [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                predicted_lights[i],
                predicted_lights[i+1],
                predicted_lights[i+2],
                1.0
            ]
        })

    # Create the light layer.
    light_layer = create_light_layer(light_data)
    layer_str = light_layer.ExportToString()

    # Write the result in the node.
    inlineusd_node = node.node('IN_RESULTS')
    inlineusd_node.parm('usdsource').set(layer_str)


def create_light_layer(light_data: list[dict]) -> Sdf.Layer:
    """Create a light layer from the given light data.

    Args:
        light_data: The light data.

    Returns:
        The light layer.
    """
    layer = Sdf.Layer.CreateAnonymous()

    for index, light_dict in enumerate(light_data):
        # Create a light in the layer.
        prim_spec = Sdf.CreatePrimInLayer(layer, f'/light{index}')
        prim_spec.typeName = 'SphereLight'
        prim_spec.specifier = Sdf.SpecifierDef

        # Set its transform.
        xform_attr_spec = Sdf.AttributeSpec(
            prim_spec, 'xformOp:transform', Sdf.ValueTypeNames.Matrix4d)
        xform_attr_spec.default = Gf.Matrix4d(*light_dict['matrix'])
        xformorder_attr_spec = Sdf.AttributeSpec(
            prim_spec, 'xformOpOrder', Sdf.ValueTypeNames.TokenArray)
        xformorder_attr_spec.default = [xform_attr_spec.name]

        # Set its attributes.
        treataspoint_attr_spec = Sdf.AttributeSpec(
            prim_spec, 'treatAsPoint', Sdf.ValueTypeNames.Bool)
        treataspoint_attr_spec.default = True
        exposure_attr_spec = Sdf.AttributeSpec(
            prim_spec, 'inputs:exposure', Sdf.ValueTypeNames.Float)
        exposure_attr_spec.default = 4.0

    return layer
