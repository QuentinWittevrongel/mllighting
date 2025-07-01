# ML Lighting

Tool to create lights from a drawing.

This project is a proof of concept and will not be maintained. Feel free to get and improve the tool on your side.


## Dependencies

- [NumPy](https://numpy.org)
- [OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO)
- [Pillow](https://python-pillow.github.io)
- [PyTorch](https://pytorch.org)
- [torchvision](https://pytorch.org/vision)


## Train

```py
python train.py DATASET_DIRECTORY OUTPUT_CHECKPOINT_FILE
```


### The dataset

The model takes 4 images to predict lights position.

```
|- dataset directory
    |- sample index
        |- albedo.png
        |- beauty.png
        |- normal.exr
        |- position.exr
        |- light.json
```

with the sample index starting at 0.

The light json contains the lights information to train the model with.

```json
[
    {
        matrix: [16 floats]
    }
]
```


## Test

```py
python test.py DATASET_DIRECTORY CHECKPOINT_FILE
```
