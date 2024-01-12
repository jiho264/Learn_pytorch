# Create ResNet32 model
##### LEE, JIHO
> Dept. of Embedded Systems Engineering, Incheon National University

> jiho264@inu.ac.kr /  jiho264@naver.com
 
- The purpose of this project is to create a ResNet32 model using Pytorch.
- The Goal of this project is that to get the accuracy of near original paper's accuracy.
- The Origin Model have 7.51% error rate in CIFAR-10 dataset.

# 0. Usage
- Run ```create_resnet.ipynb```
- The trained model is ```models/Myresnet34.pth```
---
# 1. Implementation
##### - My dev environment
- ```Ubuntu 20.04 LTS```
- ```Python 3.11.5```
- ```Pytorch 2.1.1```
- ```CUDA 11.8```
- ```pip [sklearn, copy, time]```
- ```/data/ImageNet2012/train```
- ```/data/ImageNet2012/val```
##### - Model Structure
  - Same to Origin ResNet34.
  - also, apply ```He initialization``` and ```Batch Normalization```.
##### - Preprecessing on Dataset
  - The per-pixel **mean is subtracted** from each image
  - 4 pixels are padded on each side, and **a 32 x 32 crop is randomly sampled** from the padded image or its **horizontal flip**
  - Split method
    - Train : 47.5k
    - validation : 2.5k 
      - > 현재 early stopping 말고 아무 기능 없음. 보완 필요.
    - Test : 10k
##### - Training Method
  - ```Batch size = 256```
  - ```Epochs = 5000```
    - but take early stopping when valid loss not improved within 500 epochs
  - Optimizer
    1. **```optimizer = torch.optim.Adam(model.parameters())```**
    2. ```optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)```
    3. ```optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)```
    4. ```optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)```


## The Manual from Original Paper
<details>
<summary>The Manual from Original Paper</summary>
<div markdown="1">

- Implementation :
  - [x] we initialize the weights as on He initialization
  - [x] we adopt batch normalization after each convolutional and before activation
  - [x] we use SGD with a mini-batch size of 256
  - [x] the learning rate starts from 0.1 and is divided by 10 when the error plateaus
  - [x] we use a weight decay of 0.0001 and a momentum of 0.9
  - [x] we do not use dropout
  
- In ImageNet :
  - [x] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41]. 
    > CIFAR 다룰 때엔 이미 32*32이므로 skip 
  - [x] A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. 
  - [x] The standard color augmentation in [21] is used.
    > 1. Crop + resizeing
    > 2. Principal Components Analysis (PCA 개념참고 : https://ddongwon.tistory.com/114 )
    >> 근데 ImageNet은 사이즈가 커서 괜찮은데, CIFAR는 (32,32,3)이니까 PCA 적용 안 함.

  > [21] AlexNet - Dataset section
  >> We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
  >> So we trained our network on the (centered) raw RGB values of the pixels.

- In CIFAR10 (ResNet32):
  - [x] weight decay = 0.0001 and momentum = 0.9
  - [x] adopt the weight initialization in He and BN without dropout
  - [x] (PASS) batch size = 128 on 2 GPUs
  - [x] learning rate = 0.1, divided by 10 at 32k and 48k iterations, total training iterations = 64k
    > 아니잠깐 18초씩 48k iter면 순수 train만 240시간인디..
  - [x] split to 45k/5k = train/val set
  - [x] 4 pixels are padded on each side, and a 32 x 32 crop is randomly sampled from the padded image or its horizontal flip
  - [x] the per-pixel mean is subtracted from each image
  - [x] For testing, use original images
    > Submean 안 한거로 테스트하면 완전 학습 안 되던데??????

</div>
</details>


---
# 2. Development Log
## Week 0: Summary of 2023
- [x] Setup the Leanring Process
- [x] Implementation of ResNet32 Model Structure
- [x] Implementation of the horizontal flip data augmentation in Input Dataset

## Week 1: First week, Jan, 2024
##### Goal : Getting 15% error rate in CIFAR-10 dataset.
  - [x] split to train/val and Imaplemanation of Leaning rate Decay (scheduling) 

##### My results :
  - Jan 9 
    - Crop to (32 x 32)
      - Do not apply Img resizing on CIFAR10 cuz it have already low resolution(32x32).
    - Transforms
      - transforms 들은 매 __call__마다 다시 실행되기 때문에, 랜덤 샘플링같은거 안 해도 됨.
      - transforms 는 dataset.transform.transforms.append()로 추가 가능함.
    - 1x1 conv
      - 1x1 conv에서 BN 빼면 그냥 찍는거랑 acc 똑같아져버림. 이유 잘 모르겠음.
    - Overhead 줄이려는 시도
      - float64에서 float32, float16으로 바꿔도 계산에 걸리는 시간은 비슷했으므로, float64로 진행.
  - Jan 10    
    - 내 구현과 세부적으로 다른 부분이 무엇인가? 
      > Params랑 FLOPS 보니 구현은 거의 똑같이 한 것 같은데..
      - (In ImageNet)
        - 공식 도큐멘트 수치 : num_params = 21797672, FLOPS = 3.66G
        - 내 ResNet34 수치 : num_params = 21.29M, FLOPS = 3.671G
        - https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html
      - (In CIFAR10)
        - 내 ResNet34 수치 : num_params = 21.29M, FLOPS = near 74.918M (input 2,3,32,32 일 때 0.15G)
      - (In CIFAR100)
        - 내 ResNet34 수치 : num_params = 21.336M, FLOPS = near 74.918M (input 2,3,32,32 일 때 0.15G)
      - BN 
        - affine = True가 기본인데, 이래야 gamma and beta가 학습됨.
        - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
      - ResNet34와 ResNet32는 서로 다른 것이다. 그래도 34를 같은 방식으로 코딩하고 학습 시키면 CIFAR10에서의 32의 **퍼포먼스는 분명 나올 것이다.**
      - **PCA추가하려다 CIFAR는 Low Resolution이라 적용 안 하기로 함.**
    - Adam .. !
      - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
      - optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        - 1st epoch부터 test acc 40%
        - default lr = 0.001
      - optimizer = torch.optim.Adam(model.parameters())
        - 위에거랑 비슷한듯? 미세하게 60%대 진입 빠름.
      - optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
        - 기본 Adam 호출보다 수렴이 느림. 다만 어디까지 수렴하는진 길게 학습시켜보지 않아서 모름.
    - Validation set의 용도?
      - 지금은 lr만 조정하는데 이용하고있다. 너무 낭비이지 않을까?
        > (bool) VALID : global const 
        > dataset loading할 때에 VALID에 따라 이후 training에서도 valid 제외하도록 코드 수정함.
        >> 는 이거 하려고 if문 dataloader에 몇 개 달았다가 train 시간 11s에서 22s로 늘어나서 복구함.
    
    - 하루를 마치며..
      - 하라는거 다 했는데, 왜 71%가 한계인지 잘 모르겠다. 뭘 빠트렸을까? 저자는 data augmentation에서 오히려 최신 테크닉을 사용하지 않았다.
      - Params, FLOPS로 미루어보아 ResNet34 원본과 큰 차이 없는 것 같다....
      - split ratio를 9:1말고 95:5로 변경 후 Adam으로 밤새 돌려보자
      - 2년 전에 아무렇게나 Conv 쌓아서 만든게 acc 74%네...?
        - https://github.com/jiho264/mycnn_cifar10/blob/master/mycnn_onlyconv_RGB.ipynb
      - 원본으로 학습시켜보자
        - 원본이랑 마지막 FC node 수에 따른 model크기 외, 모두 동일함.
        - Pretrained resnet34는 10여 epochs만에 약 81%까지 더 상승함.
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
        --------------------------------------------------
        ```
    - Today's Goal : Validation set을 더 적극 활용해서 acc 높여보기.  
      - Origin - CIFAR10
        > Learning rate schedualing을 위해 train을 45k/5k로 split함.
      - Origin - ImageNet에서는 
        > split하지 않고, error 평평할 때에 lr만 10으로 나눴다고 함.
          - [x] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation[41] (VGG16).
            > 굉장히 다양한 train / test image size 경우 수를 실험함.
            > 맨 처음 부분에 single scale evaluation에서 256*256을 했었기에, 다양한 crop 구현 이전에 single scale 부터 구현해보자
          - [x] A 224 x 224 crop is randomly sampled from an image or its horizontal flip, with the pre-pixel subtracted[21].
          - [ ] PCA : The standard color augmentation is used[21].
          - [ ] In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully- convolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter side is in {224, 256, 384, 480, 640}).
            >```RandomShortestSize(min_size=[224, 256, 384, 480, 640], antialias=True)```
            > test image는 shortest side를 random한 것들 중 하나로 설정하고 학습 진행 시켜보자.
            > 대신에 이미지가 정사각형이 아닌 탓에, 일부 미리보기 함수 실행 불가함. if문 처리완료.

            > 근데 standard 10 crop이 뭐야???????????

    - ImageNet2012 학습 :
      - ```num_workers=8```
      - ```pin_memory = True```
      - AMP를 train, valid, test 중 train의 forward pass에만 적용
        >```https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html```
      - Result
        <details>
        <summary>View_log</summary>
        <div markdown="1">
        
        ```
        [Epoch 1/5000] :
        100%|██████████| 5005/5005 [31:36<00:00,  2.64it/s]  
        Training time: 1896.06 seconds

        Train Loss: 5.0520 | Train Acc: 10.57%
        Valid Loss: 4.1285 | Valid Acc: 19.08%
        --------------------------------------------------
        [Epoch 2/5000] :
        100%|██████████| 5005/5005 [31:53<00:00,  2.62it/s]  
        Training time: 1913.53 seconds

        Train Loss: 3.4389 | Train Acc: 29.41%
        Valid Loss: 3.2756 | Valid Acc: 32.24%
        --------------------------------------------------
        [Epoch 3/5000] :
        100%|██████████| 5005/5005 [31:52<00:00,  2.62it/s] 
        Training time: 1912.65 seconds

        Train Loss: 2.8673 | Train Acc: 38.72%
        Valid Loss: 2.9369 | Valid Acc: 37.53%
        --------------------------------------------------
        [Epoch 4/5000] :
        100%|██████████| 5005/5005 [31:47<00:00,  2.62it/s]  
        Training time: 1907.72 seconds

        Train Loss: 2.5655 | Train Acc: 43.98%
        Valid Loss: 2.7002 | Valid Acc: 42.01%
        --------------------------------------------------
        [Epoch 5/5000] :
        100%|██████████| 5005/5005 [31:57<00:00,  2.61it/s]  
        Training time: 1917.39 seconds

        Train Loss: 2.3769 | Train Acc: 47.49%
        Valid Loss: 2.6047 | Valid Acc: 43.81%
        --------------------------------------------------
        [Epoch 6/5000] :
        100%|██████████| 5005/5005 [31:52<00:00,  2.62it/s]  
        Training time: 1912.09 seconds

        Train Loss: 2.2411 | Train Acc: 49.98%
        Valid Loss: 2.4968 | Valid Acc: 45.50%
        --------------------------------------------------
        [Epoch 7/5000] :
        100%|██████████| 5005/5005 [31:45<00:00,  2.63it/s]  
        Training time: 1905.85 seconds

        Train Loss: 2.1333 | Train Acc: 52.01%
        Valid Loss: 2.3975 | Valid Acc: 47.52%
        --------------------------------------------------
        [Epoch 8/5000] :
        100%|██████████| 5005/5005 [32:03<00:00,  2.60it/s]  
        Training time: 1923.20 seconds

        Train Loss: 2.0504 | Train Acc: 53.66%
        Valid Loss: 2.3488 | Valid Acc: 48.66%
        --------------------------------------------------
        [Epoch 9/5000] :
        100%|██████████| 5005/5005 [31:59<00:00,  2.61it/s]  
        Training time: 1919.83 seconds

        Train Loss: 1.9795 | Train Acc: 55.06%
        Valid Loss: 2.2641 | Valid Acc: 50.28%
        --------------------------------------------------
        [Epoch 10/5000] :
        100%|██████████| 5005/5005 [31:49<00:00,  2.62it/s]  
        Training time: 1909.26 seconds

        Train Loss: 1.9176 | Train Acc: 56.24%
        Valid Loss: 2.2093 | Valid Acc: 50.89%
        --------------------------------------------------
        [Epoch 11/5000] :
        100%|██████████| 5005/5005 [31:41<00:00,  2.63it/s] 
        Training time: 1901.61 seconds

        Train Loss: 1.8663 | Train Acc: 57.24%
        Valid Loss: 2.2001 | Valid Acc: 51.43%
        --------------------------------------------------
        [Epoch 12/5000] :
        100%|██████████| 5005/5005 [31:47<00:00,  2.62it/s]  
        Training time: 1907.24 seconds

        Train Loss: 1.8199 | Train Acc: 58.16%
        Valid Loss: 2.1337 | Valid Acc: 52.71%
        --------------------------------------------------
        [Epoch 13/5000] :
        100%|██████████| 5005/5005 [31:48<00:00,  2.62it/s]  
        Training time: 1908.69 seconds

        Train Loss: 1.7792 | Train Acc: 58.98%
        Valid Loss: 2.1074 | Valid Acc: 53.28%
        --------------------------------------------------
        [Epoch 14/5000] :
        100%|██████████| 5005/5005 [31:44<00:00,  2.63it/s]  
        Training time: 1904.05 seconds

        Train Loss: 1.7457 | Train Acc: 59.67%
        Valid Loss: 2.1988 | Valid Acc: 51.67%
        --------------------------------------------------
        [Epoch 15/5000] :
        100%|██████████| 5005/5005 [31:44<00:00,  2.63it/s]  
        Training time: 1904.48 seconds

        Train Loss: 1.7096 | Train Acc: 60.34%
        Valid Loss: 2.0708 | Valid Acc: 53.90%
        --------------------------------------------------
        [Epoch 16/5000] :
        100%|██████████| 5005/5005 [31:38<00:00,  2.64it/s]  
        Training time: 1898.62 seconds

        Train Loss: 1.6800 | Train Acc: 60.97%
        Valid Loss: 2.0372 | Valid Acc: 54.70%
        --------------------------------------------------
        [Epoch 17/5000] :
        100%|██████████| 5005/5005 [31:54<00:00,  2.61it/s]  
        Training time: 1914.32 seconds

        Train Loss: 1.6508 | Train Acc: 61.55%
        Valid Loss: 2.0250 | Valid Acc: 55.03%
        --------------------------------------------------
        [Epoch 18/5000] :
        100%|██████████| 5005/5005 [31:52<00:00,  2.62it/s]  
        Training time: 1912.92 seconds

        Train Loss: 1.6258 | Train Acc: 62.02%
        Valid Loss: 1.9895 | Valid Acc: 55.62%
        --------------------------------------------------
        [Epoch 19/5000] :
        100%|██████████| 5005/5005 [31:48<00:00,  2.62it/s]  
        Training time: 1908.16 seconds

        Train Loss: 1.6020 | Train Acc: 62.45%
        Valid Loss: 1.9957 | Valid Acc: 55.51%
        --------------------------------------------------
        [Epoch 20/5000] :
        100%|██████████| 5005/5005 [31:53<00:00,  2.62it/s]  
        Training time: 1913.79 seconds

        Train Loss: 1.5833 | Train Acc: 62.91%
        Valid Loss: 2.0135 | Valid Acc: 55.67%
        --------------------------------------------------
        [Epoch 21/5000] :
        100%|██████████| 5005/5005 [32:06<00:00,  2.60it/s]  
        Training time: 1926.44 seconds

        Train Loss: 1.5631 | Train Acc: 63.31%
        Valid Loss: 2.0481 | Valid Acc: 54.69%
        --------------------------------------------------
        [Epoch 22/5000] :
        100%|██████████| 5005/5005 [31:55<00:00,  2.61it/s]  
        Training time: 1915.16 seconds

        Train Loss: 1.5394 | Train Acc: 63.78%
        Valid Loss: 1.9796 | Valid Acc: 56.40%
        --------------------------------------------------
        [Epoch 23/5000] :
        100%|██████████| 5005/5005 [32:00<00:00,  2.61it/s]  
        Training time: 1920.08 seconds

        Train Loss: 1.5245 | Train Acc: 64.11%
        Valid Loss: 1.9787 | Valid Acc: 56.24%
        --------------------------------------------------
        [Epoch 24/5000] :
         58%|█████▊    | 2908/5005 [18:24<07:46,  4.50it/s]  
        ```
        </div>
        </details>


# 3. The Question
- Implementation
  - [x] Why they use stride 2 in the downsample layer? 왜 downsampling된 블럭에선 stride=2인가?
    > input은 64,8,8이고 다운 샘플 이후엔 128,4,4가 되는데, 스트레치하면서 사이즈도 줄여야 하기 때문에 stride도 2임.
  - [x] final avg pooling : 7x7x512 -> 1x1x512 이게맞나? 현재 CIFAR들은 batch*512*1*1이라 확인불가.
    > pytorch가 adoptavgpool씀.
  - [ ] 왜 제일 마지막 FC에 Relu넣으면 학습 아예 안 되지?  


# 4. Training Log
<details>
<summary>View_log</summary>
<div markdown="1">

- SGD
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
  ```
  ```
  Epoch 223/5000:
  Training time: 10.32 seconds
  Epoch 00223: reducing learning rate of group 0 to 1.0000e-02.
  Train Loss: 0.0551 | Train Acc: 98.13%
  Valid Loss: 1.9613 | Valid Acc: 64.20%
  Test Loss: 1.8799 | Test Acc: 65.19%
  --------------------------------------------------
  Epoch 424/5000:
  Training time: 10.15 seconds
  Epoch 00424: reducing learning rate of group 0 to 1.0000e-03.
  Train Loss: 0.0003 | Train Acc: 100.00%
  Valid Loss: 1.5965 | Valid Acc: 68.92%
  Test Loss: 1.5474 | Test Acc: 68.67%
  --------------------------------------------------
  Epoch 625/5000:
  Training time: 10.15 seconds
  Epoch 00625: reducing learning rate of group 0 to 1.0000e-04.
  Train Loss: 0.0003 | Train Acc: 100.00%
  Valid Loss: 1.5727 | Valid Acc: 69.04%
  Test Loss: 1.5578 | Test Acc: 68.61%
  --------------------------------------------------
  Epoch 1228/5000:
  Training time: 10.23 seconds
  Epoch 01228: reducing learning rate of group 0 to 1.0000e-07.
  Train Loss: 0.0003 | Train Acc: 100.00%
  Valid Loss: 1.5739 | Valid Acc: 68.86%
  Test Loss: 1.5707 | Test Acc: 68.46%
  --------------------------------------------------
  Epoch 2740/5000:
  Training time: 10.25 seconds
  Train Loss: 0.0003 | Train Acc: 100.00%
  Valid Loss: 1.5817 | Valid Acc: 68.84%
  Test Loss: 1.5443 | Test Acc: 68.60%
  ```
  ---
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
  scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=1000,
    factor=0.1,
    verbose=True,
    threshold=1e-5,
    min_lr=MIN_LR)
  ```
  ```
  LR decay 엄격하게 제한한 것. -> 8nn epoch까지 lr = 0.1
  [Epoch 875/5000] :
  Training time: 10.27 seconds
  Train Loss: 0.0258 | Train Acc: 99.17%
  Valid Loss: 1.3594 | Valid Acc: 74.32%
  Test  Loss: 1.3429 | Test Acc: 74.22%
  --------------------------------------------------
  ```
- adam
  ```
  [Epoch 56/5000] :
  Training time: 10.44 seconds
  Train Loss: 0.0313 | Train Acc: 98.93%
  Valid Loss: 1.6963 | Valid Acc: 71.88%
  Test  Loss: 1.6655 | Test Acc: 71.65%
  ```
  ---
- 원본이랑 마지막 FC node 수에 따른 model크기 외, 모두 동일함.
  ```
  > Original ResNet34 from torchvision.models.resnet34(pretrained=True)
  | module                 | #parameters or shape   | #flops    |
  |:-----------------------|:-----------------------|:----------|
  | model                  | 21.798M                | 0.151G    |
  |  conv1                 |  9.408K                |  4.817M   |
  |   conv1.weight         |   (64, 3, 7, 7)        |           |
  |  bn1                   |  0.128K                |  0.164M   |
  |   bn1.weight           |   (64,)                |           |
  |   bn1.bias             |   (64,)                |           |
  |  layer1                |  0.222M                |  28.557M  |
  |   layer1.0             |   73.984K              |   9.519M  |
  |    layer1.0.conv1      |    36.864K             |    4.719M |
  |    layer1.0.bn1        |    0.128K              |    40.96K |
  |    layer1.0.conv2      |    36.864K             |    4.719M |
  |    layer1.0.bn2        |    0.128K              |    40.96K |
  |   layer1.1             |   73.984K              |   9.519M  |
  |    layer1.1.conv1      |    36.864K             |    4.719M |
  |    layer1.1.bn1        |    0.128K              |    40.96K |
  |    layer1.1.conv2      |    36.864K             |    4.719M |
  |    layer1.1.bn2        |    0.128K              |    40.96K |
  |   layer1.2             |   73.984K              |   9.519M  |
  |    layer1.2.conv1      |    36.864K             |    4.719M |
  |    layer1.2.bn1        |    0.128K              |    40.96K |
  |    layer1.2.conv2      |    36.864K             |    4.719M |
  |    layer1.2.bn2        |    0.128K              |    40.96K |
  |  layer2                |  1.116M                |  35.836M  |
  |   layer2.0             |   0.23M                |   7.401M  |
  |    layer2.0.conv1      |    73.728K             |    2.359M |
  |    layer2.0.bn1        |    0.256K              |    20.48K |
  |    layer2.0.conv2      |    0.147M              |    4.719M |
  |    layer2.0.bn2        |    0.256K              |    20.48K |
  |    layer2.0.downsample |    8.448K              |    0.283M |
  |   layer2.1             |   0.295M               |   9.478M  |
  |    layer2.1.conv1      |    0.147M              |    4.719M |
  |    layer2.1.bn1        |    0.256K              |    20.48K |
  |    layer2.1.conv2      |    0.147M              |    4.719M |
  |    layer2.1.bn2        |    0.256K              |    20.48K |
  |   layer2.2             |   0.295M               |   9.478M  |
  |    layer2.2.conv1      |    0.147M              |    4.719M |
  |    layer2.2.bn1        |    0.256K              |    20.48K |
  |    layer2.2.conv2      |    0.147M              |    4.719M |
  |    layer2.2.bn2        |    0.256K              |    20.48K |
  |   layer2.3             |   0.295M               |   9.478M  |
  |    layer2.3.conv1      |    0.147M              |    4.719M |
  |    layer2.3.bn1        |    0.256K              |    20.48K |
  |    layer2.3.conv2      |    0.147M              |    4.719M |
  |    layer2.3.bn2        |    0.256K              |    20.48K |
  |  layer3                |  6.822M                |  54.659M  |
  |   layer3.0             |   0.919M               |   7.371M  |
  |    layer3.0.conv1      |    0.295M              |    2.359M |
  |    layer3.0.bn1        |    0.512K              |    10.24K |
  |    layer3.0.conv2      |    0.59M               |    4.719M |
  |    layer3.0.bn2        |    0.512K              |    10.24K |
  |    layer3.0.downsample |    33.28K              |    0.272M |
  |   layer3.1             |   1.181M               |   9.458M  |
  |    layer3.1.conv1      |    0.59M               |    4.719M |
  |    layer3.1.bn1        |    0.512K              |    10.24K |
  |    layer3.1.conv2      |    0.59M               |    4.719M |
  |    layer3.1.bn2        |    0.512K              |    10.24K |
  |   layer3.2             |   1.181M               |   9.458M  |
  |    layer3.2.conv1      |    0.59M               |    4.719M |
  |    layer3.2.bn1        |    0.512K              |    10.24K |
  |    layer3.2.conv2      |    0.59M               |    4.719M |
  |    layer3.2.bn2        |    0.512K              |    10.24K |
  |   layer3.3             |   1.181M               |   9.458M  |
  |    layer3.3.conv1      |    0.59M               |    4.719M |
  |    layer3.3.bn1        |    0.512K              |    10.24K |
  |    layer3.3.conv2      |    0.59M               |    4.719M |
  |    layer3.3.bn2        |    0.512K              |    10.24K |
  |   layer3.4             |   1.181M               |   9.458M  |
  |    layer3.4.conv1      |    0.59M               |    4.719M |
  |    layer3.4.bn1        |    0.512K              |    10.24K |
  |    layer3.4.conv2      |    0.59M               |    4.719M |
  |    layer3.4.bn2        |    0.512K              |    10.24K |
  |   layer3.5             |   1.181M               |   9.458M  |
  |    layer3.5.conv1      |    0.59M               |    4.719M |
  |    layer3.5.bn1        |    0.512K              |    10.24K |
  |    layer3.5.conv2      |    0.59M               |    4.719M |
  |    layer3.5.bn2        |    0.512K              |    10.24K |
  |  layer4                |  13.114M               |  26.25M   |
  |   layer4.0             |   3.673M               |   7.355M  |
  |    layer4.0.conv1      |    1.18M               |    2.359M |
  |    layer4.0.bn1        |    1.024K              |    5.12K  |
  |    layer4.0.conv2      |    2.359M              |    4.719M |
  |    layer4.0.bn2        |    1.024K              |    5.12K  |
  |    layer4.0.downsample |    0.132M              |    0.267M |
  |   layer4.1             |   4.721M               |   9.447M  |
  |    layer4.1.conv1      |    2.359M              |    4.719M |
  |    layer4.1.bn1        |    1.024K              |    5.12K  |
  |    layer4.1.conv2      |    2.359M              |    4.719M |
  |    layer4.1.bn2        |    1.024K              |    5.12K  |
  |   layer4.2             |   4.721M               |   9.447M  |
  |    layer4.2.conv1      |    2.359M              |    4.719M |
  |    layer4.2.bn1        |    1.024K              |    5.12K  |
  |    layer4.2.conv2      |    2.359M              |    4.719M |
  |    layer4.2.bn2        |    1.024K              |    5.12K  |
  |  fc                    |  0.513M                |  1.024M   |
  |   fc.weight            |   (1000, 512)          |           |
  |   fc.bias              |   (1000,)              |           |
  |  avgpool               |                        |  1.024K   |
    ```
  ---
  ```
  > My ResNet34
  | module                       | #parameters or shape   | #flops    |
  |:-----------------------------|:-----------------------|:----------|
  | model                        | 21.29M                 | 0.15G     |
  |  conv1                       |  9.408K                |  4.817M   |
  |   conv1.weight               |   (64, 3, 7, 7)        |           |
  |  bn1                         |  0.128K                |  0.164M   |
  |   bn1.weight                 |   (64,)                |           |
  |   bn1.bias                   |   (64,)                |           |
  |  conv64blocks                |  0.222M                |  28.557M  |
  |   conv64blocks.0             |   73.984K              |   9.519M  |
  |    conv64blocks.0.conv1      |    36.864K             |    4.719M |
  |    conv64blocks.0.bn1        |    0.128K              |    40.96K |
  |    conv64blocks.0.conv2      |    36.864K             |    4.719M |
  |    conv64blocks.0.bn2        |    0.128K              |    40.96K |
  |   conv64blocks.1             |   73.984K              |   9.519M  |
  |    conv64blocks.1.conv1      |    36.864K             |    4.719M |
  |    conv64blocks.1.bn1        |    0.128K              |    40.96K |
  |    conv64blocks.1.conv2      |    36.864K             |    4.719M |
  |    conv64blocks.1.bn2        |    0.128K              |    40.96K |
  |   conv64blocks.2             |   73.984K              |   9.519M  |
  |    conv64blocks.2.conv1      |    36.864K             |    4.719M |
  |    conv64blocks.2.bn1        |    0.128K              |    40.96K |
  |    conv64blocks.2.conv2      |    36.864K             |    4.719M |
  |    conv64blocks.2.bn2        |    0.128K              |    40.96K |
  |  conv128blocks               |  1.116M                |  35.836M  |
  |   conv128blocks.0            |   0.23M                |   7.401M  |
  |    conv128blocks.0.conv1     |    73.728K             |    2.359M |
  |    conv128blocks.0.bn1       |    0.256K              |    20.48K |
  |    conv128blocks.0.conv2     |    0.147M              |    4.719M |
  |    conv128blocks.0.bn2       |    0.256K              |    20.48K |
  |    conv128blocks.0.conv_down |    8.192K              |    0.262M |
  |    conv128blocks.0.bn_down   |    0.256K              |    20.48K |
  |   conv128blocks.1            |   0.295M               |   9.478M  |
  |    conv128blocks.1.conv1     |    0.147M              |    4.719M |
  |    conv128blocks.1.bn1       |    0.256K              |    20.48K |
  |    conv128blocks.1.conv2     |    0.147M              |    4.719M |
  |    conv128blocks.1.bn2       |    0.256K              |    20.48K |
  |   conv128blocks.2            |   0.295M               |   9.478M  |
  |    conv128blocks.2.conv1     |    0.147M              |    4.719M |
  |    conv128blocks.2.bn1       |    0.256K              |    20.48K |
  |    conv128blocks.2.conv2     |    0.147M              |    4.719M |
  |    conv128blocks.2.bn2       |    0.256K              |    20.48K |
  |   conv128blocks.3            |   0.295M               |   9.478M  |
  |    conv128blocks.3.conv1     |    0.147M              |    4.719M |
  |    conv128blocks.3.bn1       |    0.256K              |    20.48K |
  |    conv128blocks.3.conv2     |    0.147M              |    4.719M |
  |    conv128blocks.3.bn2       |    0.256K              |    20.48K |
  |  conv256blocks               |  6.822M                |  54.659M  |
  |   conv256blocks.0            |   0.919M               |   7.371M  |
  |    conv256blocks.0.conv1     |    0.295M              |    2.359M |
  |    conv256blocks.0.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.0.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.0.bn2       |    0.512K              |    10.24K |
  |    conv256blocks.0.conv_down |    32.768K             |    0.262M |
  |    conv256blocks.0.bn_down   |    0.512K              |    10.24K |
  |   conv256blocks.1            |   1.181M               |   9.458M  |
  |    conv256blocks.1.conv1     |    0.59M               |    4.719M |
  |    conv256blocks.1.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.1.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.1.bn2       |    0.512K              |    10.24K |
  |   conv256blocks.2            |   1.181M               |   9.458M  |
  |    conv256blocks.2.conv1     |    0.59M               |    4.719M |
  |    conv256blocks.2.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.2.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.2.bn2       |    0.512K              |    10.24K |
  |   conv256blocks.3            |   1.181M               |   9.458M  |
  |    conv256blocks.3.conv1     |    0.59M               |    4.719M |
  |    conv256blocks.3.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.3.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.3.bn2       |    0.512K              |    10.24K |
  |   conv256blocks.4            |   1.181M               |   9.458M  |
  |    conv256blocks.4.conv1     |    0.59M               |    4.719M |
  |    conv256blocks.4.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.4.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.4.bn2       |    0.512K              |    10.24K |
  |   conv256blocks.5            |   1.181M               |   9.458M  |
  |    conv256blocks.5.conv1     |    0.59M               |    4.719M |
  |    conv256blocks.5.bn1       |    0.512K              |    10.24K |
  |    conv256blocks.5.conv2     |    0.59M               |    4.719M |
  |    conv256blocks.5.bn2       |    0.512K              |    10.24K |
  |  conv512blocks               |  13.114M               |  26.25M   |
  |   conv512blocks.0            |   3.673M               |   7.355M  |
  |    conv512blocks.0.conv1     |    1.18M               |    2.359M |
  |    conv512blocks.0.bn1       |    1.024K              |    5.12K  |
  |    conv512blocks.0.conv2     |    2.359M              |    4.719M |
  |    conv512blocks.0.bn2       |    1.024K              |    5.12K  |
  |    conv512blocks.0.conv_down |    0.131M              |    0.262M |
  |    conv512blocks.0.bn_down   |    1.024K              |    5.12K  |
  |   conv512blocks.1            |   4.721M               |   9.447M  |
  |    conv512blocks.1.conv1     |    2.359M              |    4.719M |
  |    conv512blocks.1.bn1       |    1.024K              |    5.12K  |
  |    conv512blocks.1.conv2     |    2.359M              |    4.719M |
  |    conv512blocks.1.bn2       |    1.024K              |    5.12K  |
  |   conv512blocks.2            |   4.721M               |   9.447M  |
  |    conv512blocks.2.conv1     |    2.359M              |    4.719M |
  |    conv512blocks.2.bn1       |    1.024K              |    5.12K  |
  |    conv512blocks.2.conv2     |    2.359M              |    4.719M |
  |    conv512blocks.2.bn2       |    1.024K              |    5.12K  |
  |  fc1                         |  5.13K                 |  10.24K   |
  |   fc1.weight                 |   (10, 512)            |           |
  |   fc1.bias                   |   (10,)                |           |
  |  avgpool                     |                        |  1.024K   |

  ```

</div>
</details>

# 5. etc..
#### ImageNet Download 이후 작업
  ##### Download the ImageNet dataset
  The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

  1. Download the images from http://image-net.org/download-images

  2. Extract the training data:
      ```bash
      mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
      tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
      find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
      cd ..
      ```

  3. Extract the validation data and move images to subfolders:
      ```bash
      mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
      wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
      ```

