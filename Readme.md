# Create ResNet32 model
##### LEE, JIHO
> Dept. of Embedded Systems Engineering, Incheon National University
> jiho264@inu.ac.kr /  jiho264@naver.com
 
- The purpose of this project is to create a ResNet32 model using Pytorch.
- The Goal of this project is that to get the accuracy of near original paper's accuracy.
- The Origin Model have 7.51% error rate in CIFAR-10 dataset.

# The Manual from Original Paper
## They's Setup
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
  - [ ] The standard color augmentation in [21] is used.
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
    - ResNet34와 ResNet32는 서로 다른 것이다. 그래도 34를 같은 방식으로 코딩하고 학습 시키면 CIFAR10에서의 32의 퍼포먼스는 분명 나올 것이다.
    - PCA추가하려다 CIFAR는 Low Resolution이라 적용 안 하기로 함.
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

# The Question
- Implementation
  - [x] Why they use stride 2 in the downsample layer? 왜 downsampling된 블럭에선 stride=2인가?
    > input은 64,8,8이고 다운 샘플 이후엔 128,4,4가 되는데, 스트레치하면서 사이즈도 줄여야 하기 때문에 stride도 2임.
  - [ ] 왜 batchnorm에서 eps를 비롯한 옵션들의 설정 추가가 유효했는가? 기존엔 #value만 적었었음.
  - [x] final avg pooling : 7x7x512 -> 1x1x512 이게맞나? 현재 CIFAR들은 batch*512*1*1이라 확인불가.
    > pytorch가 adoptavgpool씀.
- Training
  - [ ] 왜 Adam에 LR 0.1을 주면 학습이 안 되는가?
  - [ ] 왜 제일 마지막 FC에 Relu넣으면 학습 아예 안 되지?


# Training Log
- SGD
  ```
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

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
  ```
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
  scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=1000,
    factor=0.1,
    verbose=True,
    threshold=1e-5,
    min_lr=MIN_LR)

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
  Epoch 20/20:
  Train Loss: 1.4475 | Train Acc: 58.53%
  Test Loss: 2.5981 | Test Acc: 37.45%
  ```