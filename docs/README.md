출처 : (https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/)

상기 출처에 code를 바탕으로 손글씨 일부만 학습시켜보았다.

Origin Image            |  GT Target Image   |  Result : 단계별 학습 진행
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-3](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0017_0205_AC00_KO_001_091028.png "Original  Image")  |  ![alt-text-4](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0017_0205_AC00_KO_001_091028_gt.png "GT Target Image")  |   ![alt-text-5](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0017_0205_AC00_KO_001_091028.gif "Result")

Origin Image            |  GT Target Image   |  Result : 단계별 학습 진행
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-3](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0018_0523_AC00_KO_001_091028.png "Original  Image")  |  ![alt-text-4](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0018_0523_AC00_KO_001_091028_gt.png "GT Target Image")  |   ![alt-text-5](https://github.com/tenjumh/Semantic-Segmentation/blob/master/Images/HIL_001_0018_0523_AC00_KO_001_091028.gif "Result")

# Semantic Segmentation Suite in TensorFlow

## News

### What's New

- This repo has been depricated and will no longer be handling issues. Feel free to use as is :)

## Description
This repository serves as a Semantic Segmentation Suite. The goal is to easily be able to implement, train, and test new Semantic Segmentation models! Complete with the following:

- Training and testing modes
- Data augmentation
- Several state-of-the-art models. Easily **plug and play** with different models
- Able to use **any** dataset
- Evaluation including precision, recall, f1 score, average accuracy, per-class accuracy, and mean IoU
- Plotting of loss function and accuracy over epochs

**Any suggestions to improve this repository, including any new segmentation models you would like to see are welcome!**

You can also check out my [Transfer Learning Suite](https://github.com/tenjumh/Transfer_Learning_Suite).

## Citing

If you find this repository useful, please consider citing it using a link to the repo :)

## Frontends

The following feature extraction models are currently made available:

- [MobileNetV2](https://arxiv.org/abs/1801.04381), [ResNet50/101/152](https://arxiv.org/abs/1512.03385), and [InceptionV4](https://arxiv.org/abs/1602.07261)

## Models

The following segmentation models are currently made available:

- [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions.

- [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions. In addition, it employs additive skip connections from the encoder to the decoder. 

- [Mobile UNet for Semantic Segmentation](https://arxiv.org/abs/1704.04861). Combining the ideas of MobileNets Depthwise Separable Convolutions with UNet to build a high speed, low parameter Semantic Segmentation model.

- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105). In this paper, the capability of global context information by different-region based context aggregation is applied through a pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet). **Note that the original PSPNet uses a ResNet with dilated convolutions, but the one is this respository has only a regular ResNet.**

- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326). Uses a downsampling-upsampling style encoder-decoder network. Each stage i.e between the pooling layers uses dense blocks. In addition, it concatenated skip connections from the encoder to the decoder. In the code, this is the FC-DenseNet model.

- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587). This is the DeepLabV3 network. Uses Atrous Spatial Pyramid Pooling to capture multi-scale context by using multiple atrous rates. This creates a large receptive field. 

- [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612). A multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions.

- [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323). Combines multi-scale context with pixel-level accuracy by using two processing streams within the network. The residual stream carries information at the full image resolution, enabling precise adherence to segment boundaries. The pooling stream undergoes a sequence of pooling operations
to obtain robust features for recognition. The two streams are coupled at the full image resolution using residuals. In the code, this is the FRRN model.

- [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719). Proposes a Global Convolutional Network to address both the classification and localization issues for the semantic segmentation. Uses large separable kernals to expand the receptive field, plus a boundary refinement block to further improve localization performance near boundaries. 

- [AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions](http://ais.informatik.uni-freiburg.de/publications/papers/valada17icra.pdf) Modifies the ResNet50 architecture by performing the lower resolution processing using a multi-scale strategy with atrous convolutions. This is a slightly modified version using bilinear upscaling instead of transposed convolutions as I found it gave better results.

- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545). Proposes a compressed-PSPNet-based image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance to address this challenge. Most of the processing is done at low resolution for high speed and the multi-scale auxillary loss helps get an accurate model. **Note that for this model, I have implemented the network but have not integrated its training yet**

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611). This is the DeepLabV3+ network which adds a Decoder module on top of the regular DeepLabV3 model.

- [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html). Combines many different scales using dilated convolution but with dense connections

- [Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/html/Bilinski_Dense_Decoder_Shortcut_CVPR_2018_paper.html). Dense Decoder Shorcut Connections using dense connectivity in the decoder stage of the segmentation model. **Note: this network takes a bit of extra time to load due to the construction of the ResNeXt blocks** 

- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897). BiSeNet use a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features while having a parallel Context Path with a fast downsampling strategy to obtain sufficient receptive field. 

- Or make your own and plug and play!


## Files and Directories


- **train.py:** Training on the dataset of your choice. Default is CamVid

- **test.py:** Testing on the dataset of your choice. Default is CamVid

- **predict.py:** Use your newly trained model to run a prediction on a single image

- **helper.py:** Quick helper functions for data preparation and visualization

- **utils.py:** Utilities for printing, debugging, testing, and evaluation

- **models:** Folder containing all model files. Use this to build your models, or use a pre-built one

- **CamVid:** The CamVid datatset for Semantic Segmentation as a test bed. This is the 32 class version

- **checkpoints:** Checkpoint files for each epoch during training

- **Test:** Test results including images, per-class accuracies, precision, recall, and f1 score


## Installation
This project has the following dependencies:

실행환경에 대해서 정리해 보자면
아나콘다로 가상환경을 구성하여 해보았다.
CUDA 9.0 cuDNN v7.0 환경으로 구성했고 많은 다양한 tensorflow gpu버전을 설치했는데 tensorflow1.9버전이 실행시 잘 진행되었다 물론 에러가 발생했지만 잘 확인해보면 해결이 된다.

>conda create -n xxx pip python=3.6

>activate xxx

>pip install tensorflow_gpu-1.9.0-cp36-cp36m-win_amd64.whl

*tensorflow_gpu-1.9.0-cp36-cp36m-win_amd64.whl은 사전에 다운로드 받아놨다. 최신(19.8월 기준 1.14.0은 CUDA10만 지원되고 그 하위 버전은 실행시 해결할 수 없는 에러들이 발생됨)

>pip install Pillow

>pip install scipy==1.1.0

*scipy 1.1.0가 아닌 최신 버전을 쓰면, imread에러가 계속발생한다.
최신 scipy에서는 imread를 지원하지 않는듯하다.

>pip install numpy

>pip install matplotlib

>pip install scikit-learn

>pip install opencv-python

이정도만 설치하면 패키지는 다 설치한 것 같음.

>python train.py

실행하면 
subprocess.CalledProcessError: Command '['python', 'utils/get_pretrained_checkpoints.py', '--model=InceptionV4']' returned non-zero exit status 1. 

에러가 발생할 수 있는데 해당 모델이 다운로드가 안되서 발생한 것일 수 있다. 해결안되면 utils/get_pretrained_checkpoints.py가서 해당 url에서 모델을 수동으로 다운로드 받아서 models 폴더에 ckpt파일을 저장해 두면 학습이 진행된다.

## Usage
The only thing you have to do to get started is set up the folders in the following structure:

├── "dataset_name"

|   ├── train

|   ├── train_labels

|   ├── val

|   ├── val_labels

|   ├── test

|   ├── test_labels


Put a text file under the dataset directory called "class_dict.csv" which contains the list of classes along with the R, G, B colour labels to visualize the segmentation results. This kind of dictionairy is usually supplied with the dataset. Here is an example for the CamVid dataset:

```
name,r,g,b
Animal,64,128,64
Archway,192,0,128
Bicyclist,0,128, 192
Bridge,0, 128, 64
Building,128, 0, 0
Car,64, 0, 128
CartLuggagePram,64, 0, 192
Child,192, 128, 64
Column_Pole,192, 192, 128
Fence,64, 64, 128
LaneMkgsDriv,128, 0, 192
LaneMkgsNonDriv,192, 0, 64
Misc_Text,128, 128, 64
MotorcycleScooter,192, 0, 192
OtherMoving,128, 64, 64
ParkingBlock,64, 192, 128
Pedestrian,64, 64, 0
Road,128, 64, 128
RoadShoulder,128, 128, 192
Sidewalk,0, 0, 192
SignSymbol,192, 128, 128
Sky,128, 128, 128
SUVPickupTruck,64, 128,192
TrafficCone,0, 0, 64
TrafficLight,0, 64, 64
Train,192, 64, 128
Tree,128, 128, 0
Truck_Bus,192, 128, 192
Tunnel,64, 0, 64
VegetationMisc,192, 192, 0
Void,0, 0, 0
Wall,64, 192, 0
```

**Note:** If you are using any of the networks that rely on a pre-trained ResNet, then you will need to download the pre-trained weights using the provided script. These are currently: PSPNet, RefineNet, DeepLabV3, DeepLabV3+, GCN.

Then you can simply run `train.py`! Check out the optional command line arguments:

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS]
                [--checkpoint_step CHECKPOINT_STEP]
                [--validation_step VALIDATION_STEP] [--image IMAGE]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                [--h_flip H_FLIP] [--v_flip V_FLIP] [--brightness BRIGHTNESS]
                [--rotation ROTATION] [--model MODEL] [--frontend FRONTEND]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --checkpoint_step CHECKPOINT_STEP
                        How often to save checkpoints (epochs)
  --validation_step VALIDATION_STEP
                        How often to perform validation (epochs)
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change
                        as a factor between 0.0 and 1.0. For example, 0.1
                        represents a max brightness change of 10% (+-).
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle in
                        degrees.
  --model MODEL         The model you are using. See model_builder.py for
                        supported models
  --frontend FRONTEND   The frontend you are using. See frontend_builder.py
                        for supported models

```
    

## Results

These are some **sample results** for the CamVid dataset with 11 classes (previous research version).

In training, I used a batch size of 1 and image size of 352x480. The following results are for the FC-DenseNet103 model trained for 300 epochs. I used RMSProp with learning rate 0.001 and decay 0.995. I **did not** use any data augmentation like in the paper. I also didn't use any class balancing. These are just some quick and dirty example results.

**Note that the checkpoint files are not uploaded to this repository since they are too big for GitHub (greater than 100 MB)**


| Class 	| Original Accuracy  	| My Accuracy |
| ------------- 		| ------------- | -------------|
| Sky  		| 93.0 | 94.1  |
| Building 		| 83.0  | 81.2  |
| Pole  		| 37.8  | 38.3  |
| Road 		| 94.5  | 97.5  |
| Pavement  		| 82.2  | 87.9  |
| Tree 		| 77.3  | 75.5  |
| SignSymbol  		| 43.9  | 49.7  |
| Fence 		| 37.1  | 69.0  |
| Car  		| 77.3  | 87.0  |
| Pedestrian 		| 59.6  | 60.3  |
| Bicyclist  		| 50.5  | 75.3  |
| Unlabelled 		| N/A  | 40.9  |
| Global  		| 91.5 | 89.6  |


Loss vs Epochs            |  Val. Acc. vs Epochs
:-------------------------:|:-------------------------:
![alt text-1](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/loss_vs_epochs.png)  |  ![alt text-2](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/accuracy_vs_epochs.png)


Original            |  GT   |  Result
:-------------------------:|:-------------------------:|:-------------------------:
![alt-text-3](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550.png "Original")  |  ![alt-text-4](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_gt.png "GT")  |   ![alt-text-5](https://github.com/GeorgeSeif/FC-DenseNet-Tiramisu/blob/master/Images/0001TP_008550_pred.png "Result")

