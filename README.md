# Care For Your Ovary

### A computer-aided diagnoisis system for ovarian tumor based on ultrasonic images


### Main Function:
- Load ultrasonic Image (Support .PNG / .JPG)
- Classification of ovarian lesions (Current models include ResNet, ResNeXt, and DenseNet)
- Visualization of the classification result
- Segmentation of ovarian lesions（Current models include UNet，DeepLabv3plus, and PSPNet)
- Save the segmentation result, segmentation mask, and visualization result

## Display
### Ovarian Lesion Segmentation
![segmentation.png](https://github.com/1024803482/CareForYourOvary/blob/master/Display/segmentation.png)
### Visualization of Lesion Classification
![classification.png](https://github.com/1024803482/CareForYourOvary/blob/master/Display/classification.png)

## Environment Setup
1. Clone the repo:
`git clone https://github.com/1024803482/CareForYourOvary` 

2. Setup environment:

    ```
    # Download libs
    pip install numpy 
    pip install PyQT5 
    pip install torch==1.8.0 torchvision==0.9.0
    pip install matplotlib
    pip install einops
    pip install segmentation_models_pytorch==0.2.1
    pip install opencv
    pip install imageio==2.9.0
    pip install PIL 
    ```
    
## Weights

The weights of classifier and segmenter can be download: 

- BaiDuYun: https://pan.baidu.com/s/1ZzAd3mvGeFx-2dJlWEDpqw , password: rc72
- Google Drive: something error.

The .EXE can be run directly:

- BaiDuYun: https://pan.baidu.com/s/17OQu5WpjSRa3bkVSO4ei1Q , password: n2o0
- Google Drive: something error.

## Note

This system is used for academic purposes, please indicate the source. We will update our system soon!

If you have any question, please discuss with me by sending email to cailh@buaa.edu.cn / ceilinghans@gmail.com.

## Citation
if you find this code helpful, please cite:

  ```
  @article{DBLP:journals/corr/abs-2207-06799,
    author    = {Qi Zhao and
                 Shuchang Lyu and
                 Wenpei Bai and
                 Linghan Cai and
                 Binghao Liu and
                 Meijing Wu and
                 Xiubo Sang and
                 Min Yang and
                 Lijiang Chen},
    title     = {A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation},
    journal   = {CoRR},
    volume    = {abs/2207.06799},
    year      = {2022},
  }
  
  @inproceedings{cai2022using,
    author = {Cai Linghan and
              Wu Meijing and 
              Chen Lijiang and
              Bai Wenpei and
              Yang Min and
              Lyu Shuchang and
              Zhao, Qi},
    title =  {Using Guided Self-Attention with Local Information for Polyp Segmentation},
    booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages = {629--638},
    year = {2022},
    organization={Springer}
}
  ```
  
## Links
- The Multi-Modality Ovarian Tumor Ultrasound Image Dataset (MMOTU): https://github.com/cv516Buaa/MMOTU_DS2Net
- The segmentation models: https://github.com/qubvel/segmentation_models.pytorch
- The classification models: https://github.com/pytorch/pytorch
