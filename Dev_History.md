# Create ResNet32 model
- The purpose of this project is to create a ResNet32 model using Pytorch.
- The Goal of this project is that to get the accuracy of near original paper's accuracy.
- The Origin Model have 7.51% error rate in CIFAR-10 dataset.

# The Manual from Original Paper
## They's Setup
- ResNet paper :
- [ ] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41]. 
- [ ] A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. 
- [ ] The standard color augmentation in [21] is used.
```
- AlexNet - Dataset section
- We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
- So we trained our network on the (centered) raw RGB values of the pixels.
```
---
- [x] we initialize the weights as on He initialization
- [x] we adopt batch normalization after each convolutional and before activation
- [x] we use SGD with a mini-batch size of 256
- [x] the learning rate starts from 0.1 and is divided by 10 when the error plateaus
- [x] we use a weight decay of 0.0001 and a momentum of 0.9
- [x] we do not use dropout
---
- [ ] It is useful when training a classification problem with C classes. 
- [ ] If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. 
- [ ] This is particularly useful when you have an unbalanced training set. 
- [ ] The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
---
- In CIFAR10 :
- [x] weight decay = 0.0001 and momentum = 0.9
- [x] adopt the weight initialization in He and BN without dropout
- [ ] batch size = 128 on 2 GPUs
- [x] learning rate = 0.1, divided by 10 at 32k and 48k iterations
  > 아니잠깐 18초씩 48k iter면 순수 train만 240시간인디..
- [ ] total training iterations = 64k
- [x] 45k/5k = train/val set
- [ ] 4 pixels are padded on each side, and a 32 x 32 crop is randomly sampled from the padded image or its horizontal flip
- [ ] the per-pixel mean is subtracted from each image
- [ ] For testing, use original images
  > Submean 안 한거로 테스트하면 완전 학습 안 되던데??????
## Week 0: Summary of 2023
- [x] Setup the Leanring Process
- [x] Implemantation of ResNet32 Model Structure
- [x] Implemantation of the horizontal flip data augmentation in Input Dataset

- problem : 
  - ```I can not get high accuracy```

## Week 1: First week, Jan, 2024
- Goal : Getting 15% error rate in CIFAR-10 dataset.
- [x] split to train/val and Imaplemanation of Leaning rate Decay (scheduling)
```
Epoch 4/20:
Training time: 17.75 seconds
Epoch 00004: reducing learning rate of group 0 to 1.0000e-02.
Train Loss: 1.4534 | Train Acc: 46.64%
Valid Loss: 1.4734 | Train Acc: 47.60%
Test Loss: 2.0706 | Test Acc: 28.00%
--------------------------------------------------
```
- My results : 
  > The Overhead for Preprecessing is not critical.
  - Do not apply Img resizing on CIFAR10 cuz it have already low resolution(32x32).
  - The Apply on half submean + Apply on half submean + horizontal flip that's train time for single epoch is 25s on CIFAR10
  - The train time for single epoch is 12.06 on CIFAR10
  
****

## The Question about Working Process of ResNet
- [ ] Why they use stride 2 in the downsample layer?
- [x] 왜 downsampling된 블럭에선 stride=2인가?
  > input은 64,8,8이고 다운 샘플 이후엔 128,4,4가 되는데, 스트레치하면서 사이즈도 줄여야 하기 때문에 stride도 2임.
- [ ] 왜 batchnorm에서 eps를 비롯한 옵션들의 설정 추가가 유효했는가? 기존엔 #value만 적었었음.
- [ ] 왜 Adam에 LR 0.1을 주면 학습이 안되는가?
- [ ] 왜 제일 마지막 FC에 Relu넣으면 학습 아예 안 되지?
- [ ] final avg pooling : 7x7x512 -> 1x1x512 이게맞나? 현재 CIFAR들은 batch*512*1*1이라 확인불가.

### Result Log
- SGD
```
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

Epoch 1/20:
Train Loss: 3.0912 | Train Acc: 25.55%
Test Loss: 2.8584 | Test Acc: 29.58%
--------------------------------------------------
Epoch 20/20:
Train Loss: 0.0016 | Train Acc: 99.96%
Test Loss: 4.2381 | Test Acc: 41.72%
--------------------------------------------------
```
- adam
```
Epoch 1/20:
Train Loss: 4.3754 | Train Acc: 5.81%
Test Loss: 3.8439 | Test Acc: 9.40%
--------------------------------------------------
Epoch 20/20:
Train Loss: 1.4475 | Train Acc: 58.53%
Test Loss: 2.5981 | Test Acc: 37.45%
--------------------------------------------------
```