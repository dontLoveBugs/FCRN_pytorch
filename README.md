# FCRN implemented in Pytorch 4.0


### Introduction
This is a PyTorch(0.4.0) implementation of [Deeper Depth Prediction with Fully Convolutional Residual Networks](http://ieeexplore.ieee.org/document/7785097/). It
can use Fully Convolutional Residual Networks to realize monocular depth prediction. Currently, we can train FCRN
using NYUDepthv2(kitti will soon!).

Note: We modify the upsample module using pixelshuffle!

### Result

#### NYU Depthv2

The code was tested with Python 3.5 with Pytorch 4.0 in 2 GPU TITAN X.  We train 30 epochs.

 Method |   rml  | rmse  | log10 | Delta1 | Delta2 | Delta3 
 :-------| :------: | :------: | :------: | :------: | :------: | :------: 
 FCRN   | 0.127  | 0.573 | 0.055 | 0.811 | 0.953 | 0.988
 FCRN_ours  | 0.177 | 0.587 | - | 0.744 | - | -
 FCRN_oursv2 | 0.175 | 0.574 | - | 0.755 | - | -
 
 FCRN_ours: we set filters group in upsample: 2x2 2x3 3x3 3x3
 
 ![avatar](https://lh3.googleusercontent.com/MeWDCgAeZQnay6zBR5TWAWG0dbIe-bduhdfpRbwrj-j9yQ3JShm9RZBzLwhfowhDOLcVwlHIprMnSDIlLQxhEjXL9_UMFXTgCITM7GzRpx7rySoF91md4Z7qvsYWJg-jdmJclSIcK5DH0pHvP0w2Q_xVXdnjHe-R5kchbKWEPSAiW1V0vK02oRZwO9nNGRYU64lyltEQcoFFZJtxRU5qAJ_Mk725oE0Jhd69namBkRrDrn0W35Gc4q4jAWf2SoYpFWvRGH_tttlvpy8o3AI8BMZqIZnDV9cWknxR7iLpcU8AQ-Ean4ekxZeQ5sR-DuWj27jQZDvZlNl_HOjIQG4-ZKvs_R_FuwseA_L6ZEtrO0vBgG_HfWANwBlQIAmbkJxq2hsWW7fxS0x6IYAzrZqKIRtzdftt3HwG6CD9PF3VWaE4G8T8-VEB-zfkoZEw4VghOhNLSYnDEVzUllGf1iJAbljbdrzi4iKGfnHrLe8xgp7uyMlqk2MzHA25zB425cL2q_9wqA0tpDT8AAWDRjzDBKUEeIzx719Qx3e7wYYoKoX3wl3U-tj6yV5V1nfrZtbXKVE9snwqQMxyeMLAhSgral5pWGE-0oElAij3YoAuLM4OiWRE03KjLyjCH3rr1yUnOl3GGte7azcSXnuCvO3E2HA=w405-h809-no)
 
 FCRN_oursv2: we set filters group in upsample: 1x1 3x3 5x5 3x3(dilation = 2)
 
 ![avatar](https://lh3.googleusercontent.com/H-TdiPbZGzs2BlrgPxFJupw3Esxi9eDyGyK-FbGDmH0lWU02_eRFBnnqomvWG4i2r053wrH6ktZ8YCMiLvUUOrWGJRyn3_Vium1Ezi97F1UXYAr8vL4aC-IQQ06FA-aVEEV_2CaUPouJ8JxGvpXdvXxSSlSo-DlLTTBBc3iYj_YwoR5XjFdYe4_HpfZqLHZ20zyeL5AVA8ryx5t7c-RcB77a25DobaAxOhi10BPgdxO5qPjbA1_YITk0g3TrTAYbfyb-YLksCYcj4bHJeYp1g4YElHzw1dgoUcH2vpGu0UxzPDAiMPZfml3TimMdFr3Tmt6TIeDyoWbkyaocZJHexbySf1piShHkMiOOpLCu_wW1meI6Wpww9d1xHkc0SNcCMzgorXNF7iK_DUgN_sud_3xFTV1Z9hBCU_u8GH5KEbLaTBf4u3Gm8TFJDjLjg-Vl0rm49drlNzKI-QZnxY8beSq3HvYWuCdq-h3ulG47FErGvVJrE92i9qzOZv6uQfHFgppIz1L7no3gmcRvxbSsQAOGgz5tTkmjJArwuqZJwMnXnMYoRVm7TGcGl5GdQ9DCHFCMp1nnL0f5KKJUmmjOaFyg9T90oKMFRaGl2eKoYSb6R8U56yYdesW29hnxTQZU-_6ejtC4PJMciuAxhnvKjjo=w633-h1264-no)

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




