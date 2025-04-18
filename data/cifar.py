from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from copy import deepcopy
import numpy as np
from config import cifar_10_root, cifar_100_root, lsuncrop_root, lsunresize_root, img_root, imgresize_root
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
import torch

class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

def subsample_dataset(dataset, idxs):

    # dataset.data = dataset.data[idxs]
    # dataset.targets = np.array(dataset.targets)[idxs].tolist()
    # dataset.uq_idxs = dataset.uq_idxs[idxs]
    new_data = []
    new_targets = []
    new_uq_idxs = []

    # 更新所有属性，保持同步
    for i, (data, target, uq_idx) in enumerate(zip(dataset.data, dataset.targets, dataset.uq_idxs)):
        if i in idxs:
            new_data.append(data)
            new_targets.append(target)
            new_uq_idxs.append(uq_idx)

    # 设置更新后的属性
    dataset.data = new_data
    dataset.targets = new_targets
    dataset.uq_idxs = new_uq_idxs
    return dataset


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset

def get_equal_len_datasets(dataset1, dataset2):

    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_cifar_10_100_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False, download=True)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_cifar_10_10_datasets(train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


def get_cifar_10_10_datasets_imgrs(train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10), balance_open_set_eval=False, split_train_val=True, seed=0):
    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    # Get testset for unknown classes
    test_dataset_unknown = ImageFolder(imgresize_root, transform=test_transform)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_cifar_10_10_datasets_imgcp(train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10), balance_open_set_eval=False, split_train_val=True, seed=0):
    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    # Get testset for unknown classes
    test_dataset_unknown = ImageFolder(img_root, transform=test_transform)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_cifar_10_10_datasets_lsunrs(train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    # train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    # test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = ImageFolder(lsunresize_root, transform=test_transform)
    # test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


def get_cifar_10_10_datasets_lsuncp(train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    # train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    # test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = ImageFolder(lsuncrop_root, transform=test_transform)
    # test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

######
    from torch.utils.data import DataLoader
    dataloaders = {}
    dataloaders['test_unknown'] = DataLoader(test_dataset_unknown, batch_size=100,
                                             shuffle=True, sampler=None, num_workers=32)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


if __name__ == '__main__':

    # x = get_cifar_10_100_datasets(None, None, balance_open_set_eval=True)
    x = get_cifar_10_10_datasets_lsuncp(None, None, split_train_val=False, balance_open_set_eval=False)

    print([len(v) for k, v in x.items()])

    debug = 0