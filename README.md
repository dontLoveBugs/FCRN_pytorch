# FCRN implemented in Pytorch 4.0


### Introduction
This is a PyTorch(0.4.1) implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/). It
can use Fully Convolutional Residual Networks to realize monocular depth prediction. Currently, we can train FCRN
using NYUDepthv2(kitti will soon!).


### Result

#### NYU Depthv2

The code was tested with Python 3.5 with Pytorch 0.4.1 in 12GB TITAN X.  We train 60 epochs with batch size = 16. The trained model can be download from [BaiduYun](https://pan.baidu.com/s/1A3lq0ntPKBOH-En818bo8A).

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 FCRN   | 0.127  | 0.573 | 0.055 | 0.811 | 0.953 | 0.988
 FCRN_ours  | 0.151 | 0.526 | 0.062 | 0.804 | 0.956 | 0.988
 
![avatar](https://lh3.googleusercontent.com/vIK8ECKDaML7IE7-khw0nlxiYrlJP_U9JzRJMAmF3qTmAE53oXIjoovs19MlEiH1y3rCcqpTfHGsd9CnIzdzu0Cr55YyOihVO_baErZia9gQWHEO5ad4Uq1lbigmu_PcvYMwZwlkuoIHlSWv5LDzFJqG39HNQUSLUt0CjXoV44QwT2In3X3in2DHu2dCp5vguCnvShqmNg67lrfkobO0rRHKodBwP-DsX5xlQ2M8skhyOU7I33JtYYP96Znq43510JkUu5nv8c_RMBGN-6t3jGNlyExVAo9cjyMQ3I_BV9A9a5jZfBnBUNG6S8y_Ngr2Jo4SSB9x3CqZ7ngtj6_WhHfYi32ASZLEIS7kY4vwx__94ZTvovizz0YM5KB1_pyBRkweSo8nWmjB84V3qtQ_17mp58pk3Nr2bYuInh4ZNhqqjT9xWaHd0Wd2M01wnAU6i935uv1W2y7VZmpxMx59ixRgjgo0ywU8ZXl7ElOonjUlF7CLQ2QKy7gqUIvd5Z95b0O4bZXO3G6agFinFUfglH4lL8CWlzvWAPETS80mA-B-_nmud1_3-j5XFmOR2KnLVLXPh_11HeDHsmykqCWpbk3S5A_3_kxehG4PvVMD2Ksm7n9i2He6xw092Kla2QClcR6WUNlwOq7PMpbRk4KC8zpG=w633-h1264-no)
 

### Installation
The code was tested with Python 3.5 with Pytorch 4.0 in 2 GPU TITAN X. 

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

3. Training

    To train NYU Depth v2, please do:
    ```Shell
    python NYUDepth_train.py
    ```

    To train it on KITTI, please do:
    ```Shell
    python KITTI_train.py
    ```




