# FCRN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/). It
can use Fully Convolutional Residual Networks to realize monocular depth prediction. Currently, we can train FCRN
using NYUDepthv2 and Kitti Odometry Dataset.


### Result

#### NYU Depthv2

The code was tested with Python 3.5 with Pytorch 0.4.1 in 12GB TITAN X.  We train 60 epochs with batch size = 16. The trained model can be download from [BaiduYun](https://pan.baidu.com/s/1A3lq0ntPKBOH-En818bo8A).

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 FCRN   | 0.127  | 0.573 | 0.055 | 0.811 | 0.953 | 0.988
 FCRN_ours  | 0.149 | 0.527 | 0.062 | 0.805 | 0.954 | 0.987
 
![Image text](https://github.com/dontLoveBugs/FCRN_pytorch/blob/master/result/nyu.png)

#### Kitti Odometry
 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 FCRN_ours  | 0.113 | 4.801 | 0.048 | 0.865 | 0.957 | 0.984
 
 ![Image text](https://github.com/dontLoveBugs/FCRN_pytorch/blob/master/result/kitti.png)

### Installation
The code was tested with Python 3.5 with Pytorch 0.4.1 in 2 GPU TITAN X. 

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

2. Configure your dataset path in "dataloaders/path.py".

3. Training

    To train NYU Depth v2, please do:
    ```Shell
    python main.py --dataset nyu
    ```

    To train it on KITTI, please do:
    ```Shell
    python main.py --dataset kitti
    ```

### NYU

```
mkdir -p data/nyu
python dataloaders/nyu_convert.py
```
