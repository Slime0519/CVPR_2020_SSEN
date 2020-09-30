# CVPR_2020_SSEN
This repository contains my implementation RefSR method proposed in   
[Robust Reference-based Super-Resolution with Similarity-Aware Deformable Convolution (CVPR 2020)](
https://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf)

## Implementation Details
The paper attached Similairity Search and Extraction Network(SSEN) to baseline, for improvement of RefSR task.

![](/Description%20image/Baseline&SSEN.png)

### Baseline
Original baseline model is network called stacked residual blocks from 
[Enhanced Deep Residual Networks for Single Image Super-Resolution (CVPR 2017)](https://arxiv.org/pdf/1707.02921.pdf),
which consist of residual blocks without BN.

### SSEN
According to original paper, SSEN extracts features from the reference images in an aligned form, matching the contents 
in the pixel space without any flow supervision.   
Authors use Deformable Convolution layers in a Sequential approch, and found three 
layers of deformable convolution are optimal structure(for best performance).

![](/Description%20image/SSEN_structure.png)

For improvement of feature extraction, SSEN choose deformable convolution kernel and designed
Dyanmic offset Estimator, addtionally.   
Dynamic offset Estimator make offset of deformable convolution architecture can be able to cover
a wide range of area.

Dynamic offset estimator contains non-local blocks, for improvement of feature extraction, too.
In paper, authors description non-local blocks in the dynamic offset estimator that the features are amplified with
attention in each level of scale.


##Dataset

| Dataset name | usage               | link                                                                   |
|--------------|---------------------|------------------------------------------------------------------------|
| CUFED        | Training/Validation | https://drive.google.com/open?id=1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I     |
| CUFED5       | Test                | https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view |

## Implementation

## To-Do list

1. Network implementation   
    1. baseline
    
        - [x] Baseline implementation(stacked residual blocks)
    2. SSEN
        - implementation deformconv
            - [x] Study code of Deformable Convolutional Network
            - [x] implementation Dynamic Offset Estimator(DOE)
                + implementation non-local block
            - [x] combine two module(DOE, original DeformConv)
        - make entire SSEN structure
            - [x] connect with feature extractor in baseline
            - [x] append deformconv blocks sequentially
    3. connect two network
        - [x] summary two models  using pysummary library
        - [x] attach SSEN to Baseline of RefSR network
    
2. Get dataset and preprocessing
    - [x] get CUFED dataset for training
    - [x] get CUFED5 dataset for test
    - [x] implementation Dataloader for each tasks(training, vaild, evaluation)
        - apply random 90 degree rotation for augmentation 
        - scaling factor : 4
    
3. Training & Test
    - [x] implementation Training code
        - using ADAM optimizer 
        - lr : 1e^-4
        - b1,b2 = 0.9, 0.999
        - batch size : 32
        - epochs : 100k
        - lr scheduling : consine learning rate schedule, gamma = 0.9
    - [x] implementation evaluation code
   
4. Additional Task
    - [ ] Attach GAN module(PatchGAN)


## References
1. [Robust Reference-based Super-Resolution with Similarity-Aware Deformable Convolution (CVPR 2020)](
https://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf
)
2. [Deformable Convolutional Networks (CVPR 2017)](https://arxiv.org/pdf/1703.06211.pdf)
3. [Non-local Neural Networks (CVPR 2018)](https://arxiv.org/pdf/1711.07971.pdf)
4. [Image Super-Resolution by Neural Texture Transfer (CVPR 2019)](https://arxiv.org/pdf/1903.00834.pdf)
5. [Enhanced Deep Residual Networks for Single Image Super-Resolution (CVPR 2017)](https://arxiv.org/pdf/1707.02921.pdf)
6. [TDAN: Temporally Deformable Alignment Network for Video Super-Resolution (CVPR 2018)](https://arxiv.org/pdf/1812.02898.pdf)
7. [EDVR: Video Restoration with Enhanced Deformable Convolutional Networks (CVPR 2019)](https://arxiv.org/pdf/1905.02716.pdf)
