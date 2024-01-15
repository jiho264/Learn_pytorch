# Create ResNet
##### LEE, JIHO
> Dept. of Embedded Systems Engineering, Incheon National University

> jiho264@inu.ac.kr /  jiho264@naver.com
 
- The purpose of this project is to create a ResNet34 model using Pytorch.
- The Goal of this project is that to get the accuracy of near original paper's accuracy.
- The Origin ResNet32 have 7.51% top-1 error rate in CIFAR-10 dataset.
- The Origin ResNet34 have 21.53% top-1 error rate in ImageNet2012 dataset.
---
### Todo : 
- [ ] optimazer setting 재확인하기. 현재 Adam의 기본옵션임.
- [ ] LR Schedualer 재확인하기. 현재 Adam 이용 중이라 적용하지 않고 있음.
- [ ] Early Stopping 알고리즘 최적화. 현재 30회 valid loss 향상 없으면 마침.
---
# 1. Usage
## 1.1. Requierments
  - ```Ubuntu 20.04 LTS```
  - ```Python 3.11.5```
  - ```Pytorch 2.1.1```
  - ```CUDA 11.8```
  - ```pip [sklearn, copy, time, tqdm, matplotlib]```
  - ```/data/ImageNet2012/train```
  - ```/data/ImageNet2012/val```
## 1.2. How to run 
  - Run ```create_resnet.ipynb```
  - Options
    - ```BATCH = 256```
    - ```DATASET = {"CIFAR10", "CIFAR100", "ImageNet2012"}```
  - The trained model is ```models/Myresnet34.pth```
## 1.3. The Manual from Original Paper
### 1.3.1. Implementation about training process :
  - [x] We initialize the weights as on **He initialization**
  - [x] We adopt **batch normalization** after each convolutional and before activation
  - [x] We use **SGD** with a **mini-batch size of 256**
  - [x] The learning rate starts from **0.1** and is **divided by 10** when the error plateaus
  - [x] We use a **weight decay of 0.0001** and a **momentum of 0.9**
  - [x] We **do not use** dropout
  
### 1.3.2. ```MyResNet34``` preprocessing for ImageNet2012 :
  - [x] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41]. 
  - [x] A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. 
    > [21] AlexNet - Dataset section
    >> We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
    >> So we trained our network on the (centered) raw RGB values of the pixels.
  - [x] The standard color augmentation in [21] is used.
    > 1. Crop + resizeing
    > 2. Principal Components Analysis (PCA 개념참고 : https://ddongwon.tistory.com/114 )
    > AutoAugment로 해결
  - [ ] In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully- convolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter side is in {224, 256, 384, 480, 640}).

### 1.3.3. ```MyResNet_CIFAR``` preprocessing for CIFAR10 :
  - [x] 45k/5k train/valid split from origin train set(50k)
  - [x] 4 pixels are padded on each side, and a 32 x 32 crop is randomly sampled from the padded image or its horizontal flip.
  - [x] For testing, use original images
---

# 2. Development Log
- Jan 10    
  - 거의 모든 사항 구현.
  - BN 
    - affine = True가 기본인데, 이래야 gamma and beta가 학습됨.
    - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

- Jan 11
  - Origin과의 비교 결과
    - 같은 학습 method에서는 거의 동일한 Convergence 보임.
    - 500 epochs에서도 valid loss 줄지 않으면, 그간 최소 loss였던 weight를 저장함. -> 579에서 종료면 실제 학습은 79에서 제일 잘 나왔던 것.
      > 학습 방법의 차이에서 80%대 달성이 좌우되는 듯 하다.
    - MyResNet34
      ```
      [Epoch 580/5000] :
      Training time: 15.96 seconds
      Train Loss: 0.0000 | Train Acc: 100.00%
      Valid Loss: 2.5979 | Valid Acc: 75.64%
      Test  Loss: 2.5924 | Test Acc: 75.46%
      Early stopping after 579 epochs without improvement.
      ```
    - Origin ResNet34 (pretrained = False)
      ```
      [Epoch 506/5000] :
      Training time: 11.10 seconds
      Train Loss: 0.0040 | Train Acc: 99.89%
      Valid Loss: 2.4615 | Valid Acc: 74.32%
      Test  Loss: 2.3427 | Test Acc: 75.67%
      Early stopping after 505 epochs without improvement.
      ```
    - Origin ResNet34 (pretrained = True)
      ```
      [Epoch 134/5000] :
      Training time: 11.12 seconds
      Train Loss: 0.0043 | Train Acc: 99.86%
      Valid Loss: 1.1660 | Valid Acc: 81.80%
      Test  Loss: 1.2111 | Test Acc: 81.40%
      ```
  - ImageNet2012 학습 :
    - AMP를 train, valid, test 중 train의 forward pass에만 적용
      - https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html
    - Result
      ```
      [Epoch 23/5000] :
      100%|██████████| 5005/5005 [32:00<00:00,  2.61it/s]  
      Training time: 1920.08 seconds
      Train Loss: 1.5245 | Train Acc: 64.11%
      Valid Loss: 1.9787 | Valid Acc: 56.24%
       ```
- Jan 12 :
  - ResNet 32 추가 (n에 따라 가변적으로 ResNet 생성 가능.) 
    - Results :
      - optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001), early stop threshold 30
        ```
        [Epoch 112/5000] :
        100%|██████████| 176/176 [00:06<00:00, 25.87it/s]
        Train Loss: 0.0004 | Train Acc: 96.50%
        Valid Loss: 1.4764 | Valid Acc: 72.50%
        Test  Loss: 1.5016 | Test Acc: 72.05%
        Early stopping after 111 epochs without improvement.
        ```
      - optimizer = torch.optim.Adam(model.parameters()), early stop threshold 30
        ```
        [Epoch 109/5000] :
        100%|██████████| 176/176 [00:06<00:00, 25.57it/s]
        Train Loss: 0.0000 | Train Acc: 100.00%
        Valid Loss: 1.5166 | Valid Acc: 79.00%
        Test  Loss: 1.5319 | Test Acc: 78.46%
        Early stopping after 108 epochs without improvement.
        ```
  - amp on/off 추가. ImageNet2012 학습하는 ResNet34일 때만 적용하도록 바꿈.
- Jan 13 : 
  - ResNet32 for CIFAR10
    - Setup 1
      - train만 전처리 하고, valid, test에 ToTensor()만 적용시 507 epoch에서 stop되었고 acc 각각 100%, 80%, 58%로 나타남.
      - on CIFAR10에서 testing시, origin image 32x32x3 썼다고했는데, submean을 하지 않고서는 도저히 이렇게 나오지 않는다. 
      - Submean하는게 맞는 것 같다.
    - Setup 2
      - valid, test 모두 Submean 적용
      - 509 epochs test acc 75.42%
- Jan 15 : 
  - build training logging process 
---
# 3. Training Log
- ImageNet2012
  - Adam default
    ```
    [Epoch 1/5000] :
    100%|██████████| 5005/5005 [31:36<00:00,  2.64it/s]  
    Training time: 1896.06 seconds

    Train Loss: 5.0520 | Train Acc: 10.57%
    Valid Loss: 4.1285 | Valid Acc: 19.08%
    --------------------------------------------------
    [Epoch 23/5000] :
    100%|██████████| 5005/5005 [32:00<00:00,  2.61it/s]  
    Training time: 1920.08 seconds

    Train Loss: 1.5245 | Train Acc: 64.11%
    Valid Loss: 1.9787 | Valid Acc: 56.24%
    ```
- CIFAR10
  - SGD
    ```python
    dd
    ```
    ```
    dd
    ```
  - adam
    ```
    dd
    ```
