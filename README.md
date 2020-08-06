# CVPR_2020_SSEN
This repository contains my implementation RefSR method proposed in   
[Robust Reference-based Super-Resolution with Similarity-Aware Deformable Convolution (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf)

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

