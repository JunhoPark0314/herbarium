# The Herbarium 2021: Half-Earth Challenge - FGVC8

This project host the code for implementing hierarchical classification for Herbarium dataset.

Implementation based on Detectron2

## Dependency

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.2 ~ 1.7.1 
- [Torchvision] 0.4.0 ~ 0.8.2 depending on the version of torch.
- [scikit-learn](https://scikit-learn.org/stable/)

## Installation

```
python setup.py build develop
```

## Training

```
python train.py --config-file configs/Base-SimpleNet.yaml
```

You can adjust parameters for training by changing configs/Base-SimpleNet.yaml

