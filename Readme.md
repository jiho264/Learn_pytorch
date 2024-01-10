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
  - [ ] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41]. 
  - [ ] A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. 
  - [ ] The standard color augmentation in [21] is used.

  > [21] AlexNet - Dataset section
  >> We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
  >> So we trained our network on the (centered) raw RGB values of the pixels.

- In CIFAR10 :
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
- [x] Implemantation of ResNet32 Model Structure
- [x] Implemantation of the horizontal flip data augmentation in Input Dataset

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
****

## The Question about Working Process of ResNet
- Implementation
  - [x] Why they use stride 2 in the downsample layer? 왜 downsampling된 블럭에선 stride=2인가?
    > input은 64,8,8이고 다운 샘플 이후엔 128,4,4가 되는데, 스트레치하면서 사이즈도 줄여야 하기 때문에 stride도 2임.
  - [ ] 왜 batchnorm에서 eps를 비롯한 옵션들의 설정 추가가 유효했는가? 기존엔 #value만 적었었음.
  - [x] final avg pooling : 7x7x512 -> 1x1x512 이게맞나? 현재 CIFAR들은 batch*512*1*1이라 확인불가.
    > pytorch가 adoptavgpool씀.
- Training
  - [ ] 왜 Adam에 LR 0.1을 주면 학습이 안 되는가?
  - [ ] 왜 제일 마지막 FC에 Relu넣으면 학습 아예 안 되지?


### Result Log
- SGD
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
- adam
  ```
  Epoch 20/20:
  Train Loss: 1.4475 | Train Acc: 58.53%
  Test Loss: 2.5981 | Test Acc: 37.45%
  ```