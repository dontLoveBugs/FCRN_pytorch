# FCRN implemented in Pytorch 4.0


### Introduction
This is a PyTorch(0.4.0) implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/). It
can use Fully Convolutional Residual Networks to realize monocular depth prediction. Currently, we can train FCRN
using NYUDepthv2(kitti will soon!).

Note: We modify the upsample module in "model.py" using the upsample strategy in [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158v2), and the original upsample methods (deconv, upconv and up_proj) is available in "FCRN.py".

### Result

#### NYU Depthv2

The code was tested with Python 3.5 with Pytorch 4.0 in 2 GPU TITAN X.  We train 30 epochs.

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 FCRN   | 0.127  | 0.573 | 0.055 | 0.811 | 0.953 | 0.988
 FCRN_ours  | 0.177 | 0.587 | - | 0.744 | 0.936 | 0.984
 FCRN_oursv2 | 0.159 | 0.546 | 0.066 | 0.779 | 0.951 | 0.988
 
 FCRN_ours: we set filters group in upsample: 2x2 2x3 3x3 3x3
 
 ![avatar](https://lh3.googleusercontent.com/aJxtuQH8CVeXXQtPXUQYYjEQS_dhp8pJjVHMNSWlK0Cp4PhM_yREpB7UsfSq7onPVMe_0xbhvz6Vd9MUTkj1aGPL-uDykU8PxTELjxzbyQZv4xsEBjmwGwa6H4VQE2p3rKhSz3K1KRq8OU4v3fZ2hgeV_CBBuEPE71Lk0lo3BRFb94HDAhAqDmPQVXeKFWDP1CYmI00bLl4620As4HzvZxB-eR6yVCJJbWIlOF4v-d-9UvSoGzRYyTL1DDVvDT7308OT04XOVY7nqlrYa1naCGgyMMLwBQJcg61yPVosYA1UP6cWm4pAHRsvkT1EU1YQlhe_1-FWz1ZjZWClGBp6IFN0fjyxEbn69pHieObaX8yfS1K8NU71FMY6uIkrgz87n7CDf6fwUm4tDLtNgc7dliYIvLBGaq9C_qXbriO_FlheWS84GJfoUC1RyNyLtXQmEIdja3ZRMUhEcntsVOzFnSOvss2VDI0DWtz8QfdPpzsccuCcIM0Lv44X3MWPdXTHyVu7rtaW2Joo6QyB_qUkmrKhAGh9oJlLx3yqthEsrz5fR4cJpZr0icKEb0vygFCr9jBuRgbb7Xi6pimrHP_2RzbapgjRAW4cLm7lD5D_KrAPYYiC2g4EBvhy4r7n2HaPopxAmb7MHK33VMrFbjiKJSw=w633-h1264-no)
 
 FCRN_oursv2: we set filters group in upsample: 1x1 3x3 5x5 3x3(dilation = 2)
 
 ![avatar](https://lh3.googleusercontent.com/1TYkjPPdh4_Gelnja1hD9n5HPXr8bmuYqQRYNy9ePzSCe2P0h3_zxyVvMAhWvKKCUJktbHE63P0ELuVMLk7L1BWV8X2M0hhGN5uBN38BiLBISKecrKyozjEjeccsYn63pU7b9ie3EiT1YLiCrNSD5JU6kPMwEixz4Dg7bNDXi43tubFD5G-RbRLaW16MGU9Wmn0bzF2UEhcND_rkKH_oNhehcQOM437cLjnL_BislzeTHb69DJtVUI3vaNCZt6ORLP-vbvilSN2TAhQsW-efBqy3IcDipUOIDR_ErUpfla95T1ZWKSZCpGZ-WBd8t1pMhfXaXEbv1k0vt3PBuxl1Hhf5-rhbF1fQzXCSpboUXAN8WnGBLKQLdEk2C2DYIbNJ69Z0xCdQ1GcYtJCkGFe2YPSM1ekc-d4R9pylf6ydb42fSYvRXhamKUXP0E1Si77hvAnIG5pEjBiYcG5fPXx3HV07LnLQSSs_BOG_sEpLe0S9F2F07yX_QohiAXSMRzySkm5DXOaumaiziptyPSG9ncc7Zv443_8_8ukQtKxSczZEBRZkYR7MiOknr2doTfobvHEzvocgkV0WMFTbt_4vU3H3tKPh8ivnmLN5Hg0oyY6bnUokCV4bF9E-2BEHJ52lT3UsJbcXmU_4JzoBniBA2J8=w633-h1264-no)

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

3. You can train deeplab v3+ using xception or resnet as backbone.

    To train NYU Depth v2, please do:
    ```Shell
    python NYUDepth_train.py
    ```

    To train it on KITTI, please do:
    ```Shell
    python KITTI_train.py
    ```




