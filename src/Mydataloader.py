import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms.v2 import (
    ToTensor,
    RandomHorizontalFlip,
    Compose,
    RandomCrop,
    RandomShortestSize,
    AutoAugment,
    Normalize,
    TenCrop,
    CenterCrop,
    Pad,
    Resize,
)
from torchvision.transforms.autoaugment import AutoAugmentPolicy


class Submean(torch.nn.Module):
    # Subtract the mean from each pixel along each channel
    def __init__(self):
        super().__init__()
        return None

    def __call__(self, tensor):
        _mean = tensor.mean(axis=(1, 2))
        tensor = tensor - _mean[:, None, None]

        return tensor


class LoadDataset:
    """
    input :
        - root : datasets folder path
        - seceted_data :
            - "CIFAR10" or "CIFAR100" : Load ~ from torchvision.datasets
            - "ImageNet2012" : Load ~ from Local
    pre-processing:
        - CIFAR10, CIFAR100 :
            - Option : split train/valid with (split_ratio):(1-split_ratio) ratio (default split_ratio = 0)
            - train :
                - ToTensor(),
                - Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[1, 1, 1],inplace=True,),
                - AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                - RandomCrop(size=32, padding=4,fill=0,padding_mode="constant",),
                - RandomHorizontalFlip(self.Randp),
            - valid, test :
                - ToTensor(),
        - ImageNet2012 :
            - train :
                - RandomShortestSize(min_size=range(256, 480), antialias=True),
                - RandomCrop(size=224),
                - AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                - RandomHorizontalFlip(self.Randp),
                - ToTensor(),
                - Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
            - valid (center croped valid set) :
                - RandomShortestSize(min_size=range(256, 480), antialias=True),
                - CenterCrop(size=368),
                - ToTensor(),
                - Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
            - test (10-croped valid set):
                ToTensor(),
                TenCrop()
                ...
                ...
    output :
        - self.train_data
        - self.valid_data (default : None)
        - self.test_data (default : None)
        - num of classes
    """

    def __init__(self, root, seceted_dataset, split_ratio=0):
        self.Randp = 0.5
        self.dataset_name = seceted_dataset
        self.split_ratio = split_ratio

        if self.dataset_name[:5] == "CIFAR":
            dataset_mapping = {
                "CIFAR100": datasets.CIFAR100,
                "CIFAR10": datasets.CIFAR10,
            }
            cifar_default_transforms = Compose(
                [
                    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                    RandomCrop(
                        size=32,
                        padding=4,
                        fill=0,
                        padding_mode="constant",
                    ),
                    RandomHorizontalFlip(self.Randp),
                    ToTensor(),
                    # exject mean and std
                    # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
                    # std=1로 하면 submean
                    # https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Normalize.html#torchvision.transforms.v2.Normalize
                    Normalize(
                        mean=[0.49139968, 0.48215827, 0.44653124],
                        std=[1, 1, 1],
                        inplace=True,
                    ),
                    # Submean(),
                ],
            )
            """CIFAR10, CIFAR100에서는 ref_train에 split ratio대로 적용해서 잘라냄."""
            ref_train = dataset_mapping[self.dataset_name](
                root=root,
                train=True,
                download=False,
                transform=cifar_default_transforms,
            )
            self.test_data = dataset_mapping[self.dataset_name](
                root=root,
                train=False,
                download=False,
                transform=ToTensor(),
            )

            if self.split_ratio != 0:
                # Split to train and valid set
                total_length = len(ref_train)
                train_length = int(total_length * self.split_ratio)
                valid_length = total_length - train_length
                self.train_data, self.valid_data = random_split(
                    ref_train, [train_length, valid_length]
                )
                # Apply transform at each dataset
                self.train_data.transform = copy.deepcopy(cifar_default_transforms)
                self.valid_data.transform = ToTensor()

                self.train_data.classes = ref_train.classes
                self.valid_data.classes = ref_train.classes

                self.train_data.class_to_idx = ref_train.class_to_idx
                self.valid_data.class_to_idx = ref_train.class_to_idx

            else:
                self.train_data = ref_train
                # self.train_data.transform = copy.deepcopy(cifar_default_transforms)
                # self.train_data.transform = ToTensor()
                self.valid_data = None

            #######################################################

        elif self.dataset_name == "ImageNet2012":
            self.ImageNetRoot = "data/" + self.dataset_name + "/"

            self.train_data = datasets.ImageFolder(
                root=self.ImageNetRoot + "train",
                transform=Compose(
                    [
                        RandomShortestSize(
                            min_size=range(256, 480), antialias=True
                        ),  # 만약 이거보다 작으면 적용 안 된 작은 사이즈
                        RandomCrop(size=224),
                        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                        RandomHorizontalFlip(self.Randp),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
                        ),
                    ]
                ),
            )
            """논문에 제시된 in testing, 10crop + 멀티스케일"""
            self.valid_data = datasets.ImageFolder(
                root=self.ImageNetRoot + "val",
                transform=Compose(
                    [
                        RandomShortestSize(
                            min_size=range(256, 480), antialias=True
                        ),  # 만약 이거보다 작으면 적용 안 된 작은 사이즈
                        # VGG에서 single scale로 했을 때는 두 range의 median 값으로 crop함.
                        CenterCrop(size=368),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
                        ),
                    ]
                ),
            )
            """
            각 지정된 스케일에 따라 10 crop해야하는데, 5개 scale들의 평균을 내야하니까 좀 번거로움.
            그치만, 학습 중엔 center crop으로 eval하니, 지금 당장 필요하지는 않음.
            """
            # self.test_data = datasets.ImageFolder(
            #     root=self.ImageNetRoot + "val",
            #     transform=Compose(
            #         [
            #             RandomShortestSize(
            #                 min_size=range[224, 256, 384, 480, 640], antialias=True
            #             ),
            #             TenCrop(size=368),
            #             ToTensor(),
            #             Normalize(
            #                 mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
            #             ),
            #         ]
            #     ),
            # )
            self.test_data = None
            # tmp = [None, None, None, None, None]
            # tmp[0], tmp[1], tmp[2], tmp[3], tmp[4] = random_split(
            #     ref_test_data, [0.2, 0.2, 0.2, 0.2, 0.2]
            # )
            # self.test_data = torch.utils.data.ConcatDataset([tmp])

            # scale = [224, 256, 384, 480, 640]
            # for i in range(len(tmp)):
            #     self.test_data.datasets[i].transform = copy.deepcopy(
            #         ref_test_data.transform
            #     )
            #     self.test_data.datasets[i].transform.transforms.append(
            #         RandomShortestSize(min_size=scale[i], antialias=True)
            #     )
            #     self.test_data.datasets[i].transform.transforms.append(
            #         Pad(padding=int(scale[i] / 8), padding_mode="constant")
            #     )
            #     self.test_data.datasets[i].transform.transforms.append(
            #         TenCrop(size=scale[i])
            #     )

            #     self.test_data.datasets[i].classes = self.train_data.classes
            #     self.test_data.datasets[i].class_to_idx = self.train_data.class_to_idx

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return

    def Unpack(self, print_info=True):
        if print_info == True:
            print(
                "-----------------------------------------------------------------------"
            )
            print("Dataset : ", self.dataset_name)
            print("- Length of Train Set : ", len(self.train_data))
            if self.valid_data != None:
                print("- Length of Valid Set : ", len(self.valid_data))
            if self.test_data != None:
                print("- Length of Test Set : ", len(self.test_data))
            print("- Count of Classes : ", len(self.train_data.classes))
            print(
                "-----------------------------------------------------------------------"
            )
        return (
            self.train_data,
            self.valid_data,
            self.test_data,
            len(self.train_data.classes),
        )
