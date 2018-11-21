# FCRN implemented in Pytorch 4.0


### Introduction
This is a PyTorch(0.4.0) implementation of [FCRN](http://ieeexplore.ieee.org/document/7785097/). It
can use Fully Convolutional Residual Networks to realize monocular depth prediction. Currently, we can train FCRN
using NYUDepthv2(kitti will soon!).

Note: We modify the upsample module using pixelshuffle!


### Installation
The code was tested with Python 3.5 with Pytorch 4.0 in 2 GPU TITAN X. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone git@github.com:dontLoveBugs/FCRN_pyotrch.git
    cd FCRN_pytorch
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX
    ```

2. Configure your dataset path in code.

3. You can train deeplab v3+ using xception or resnet as backbone.

    To train NYU Depth v2, please do:
    ```Shell
    python NYUDepth_train.py
    ```

    To train it on KITTI, please do:
    ```Shell
    python KITTI_train.py
    ```




