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
    RandomResizedCrop,
    RandomShortestSize,
    AutoAugment,
    Normalize,
    Pad,
    TenCrop,
    FiveCrop,
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
        - root : "data"
        - seceted_data :
            - "CIFAR10" or "CIFAR100" : Load ~ from torchvision.datasets
            - "ImageNet2012" : Load ~ from Local
    pre-processing:
        - CIFAR10, CIFAR100 :
            - split train/valid with 9:1 ratio
            - train :
                ToTensor(),
                Random Horizontal Flip (p = 0.5),
                4 pixel zero padding and crop to (32,32,3),
                Submean(),
            - valid, test :
                ToTensor(),
                Submean(),
        - ImageNet2012 :
            - train :
                ToTensor(),
                RandomShortestSize(min_size=256, max_size=480, antialias=True),
                RandomResizedCrop([224, 224], antialias=True),
                RandomHorizontalFlip(self.Randp),
                Submean(),
            - valid :
                ToTensor(),
                RandomShortestSize(min_size=[224, 256, 384, 480, 640], antialias=True),
                RandomResizedCrop([224, 224], antialias=True),
                Submean(),
    output :
        - self.train_data
        - self.valid_data
        - self.test_data
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
                    ToTensor(),
                    # Normalize(
                    #     mean=[0.49139968, 0.48215827, 0.44653124],
                    #     std=[1, 1, 1],
                    #     inplace=True,
                    # ),
                    Submean(),
                    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                    Pad(padding=4, fill=0, padding_mode="constant"),
                    RandomResizedCrop(size=[32, 32], antialias=True),
                    RandomHorizontalFlip(self.Randp),
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
                        ToTensor(),
                        Submean(),
                        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                        RandomShortestSize(min_size=256, max_size=480, antialias=True),
                        RandomResizedCrop([224, 224], antialias=True),
                    ]
                ),
            )
            """[21]AlexNet의 Single Scale Evaluation을 valid set으로 사용"""
            self.valid_data = datasets.ImageFolder(
                root=self.ImageNetRoot + "val",
                transform=Compose(
                    [
                        ToTensor(),
                        RandomShortestSize(min_size=368, max_size=368, antialias=True),
                        # 368 / 8 = 46
                        Pad(padding=46, fill=0, padding_mode="constant"),
                        RandomResizedCrop([368, 368], antialias=True),
                    ]
                ),
            )
            """논문에 제시된 in testing, 10crop + 멀티스케일"""
            self.test_data = None
            # ref_test_data = datasets.ImageFolder(
            #     root=self.ImageNetRoot + "val",
            #     transform=Compose(
            #         [
            #             ToTensor(),
            #             Submean(),
            #         ]
            #     ),
            # )

            # _test_data = [None, None, None, None, None]

            # _scale = [224, 256, 384, 480, 640]
            # for i in range(len(_scale)):
            #     print(i)
            #     _test_data[i] = copy.deepcopy(ref_test_data)
            # """
            # padding은 이미지 크기의 1/8주고, 10crop
            # """

            # self.test_data = torch.utils.data.ConcatDataset([_test_data])
            # self.test_data.classes = self.train_data.classes
            # self.test_data.class_to_idx = self.train_data.class_to_idx

            # for i in range(len(_scale)):
            #     self.test_data.datasets[i] = copy.deepcopy(ref_test_data)
            #     self.test_data.datasets[i].transform.transforms.append(
            #         RandomShortestSize(min_size=_scale[i], antialias=True)
            #     )
            #     self.test_data.datasets[i].transform.transforms.append(
            #         PaddingWithRandomResizedCrop(_scale[i] / 8, [_scale[i], _scale[i]])
            #     )
            #     self.test_data.datasets[i].transform.transforms.append(
            #         TenCrop(_scale[i])
            #     )
            # """
            # padding은 이미지 크기의 1/8주고, 10crop
            # """

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
            if self.split_ratio != 0 or self.dataset_name == "ImageNet":
                print("- Length of Valid Set : ", len(self.valid_data))
            if self.dataset_name == "ImageNet":
                pass
            else:
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
