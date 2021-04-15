# SW_ModelCompression

This is the Pytorch implementation for "[A general model compression method for image restoration network](https://doi.org/10.1016/j.image.2021.116134)".

### SWConv

The *SWConv* is depicted as follows:

![image](figures/SWConv.pdf)

### SWConv-F

The *SWConv-F* is depicted as follows:

![image](figures/SWConv-F.pdf)

### SW-Models

The SW-DnCNN, SW-RCAN and SW-DnCNN-3 are implemented in SWDnCNN.py, SWRCAN.py and SWDnCNN3.py, respectively. The original networks can be replaced by the corresponding ones for retraining and test.