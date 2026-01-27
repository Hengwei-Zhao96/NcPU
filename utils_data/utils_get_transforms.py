import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torchvision.transforms as transforms


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    return img.rotate(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):
    assert 0.0 <= v <= 0.5
    if v <= 0.0:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l


class RandomAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)
        return img

class NumpyHWC2TensorCHW:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))
                return torch.from_numpy(img).to(dtype=torch.float)
            else:
                raise ValueError("Input numpy array must have 3 dimensions (H, W, C)")
        else:
            raise TypeError("Input must be a numpy ndarray")


def get_transforms(dataset_name):
    
    if dataset_name in ["cifar10", "cifar100"]:
        img_size = 32
    elif dataset_name in ["eurosat", "guatemala-volcano", "hurricane-florence", "hurricane-harvey", "hurricane-matthew", "hurricane-michael", "mexico-earthquake", "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire", "socal-fire", "xbd-all"]:
        img_size = 64
    elif dataset_name =="stl10":
        img_size = 96
    elif dataset_name == "abcd":
        img_size = 128

    if dataset_name in ["cifar10", "eurosat"]:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name == "stl10":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif dataset_name == "abcd":
        mean = (119.9391, 114.9877, 103.1147, 140.6345, 142.3115, 130.3468)
        std = (69.6637, 66.0012, 65.2355, 35.8397, 32.8773, 33.7203)
    elif dataset_name == "guatemala-volcano":
        mean = (74.82595670021186, 99.03509211136122, 89.08770441604872)
        std = (52.75662590592767, 46.766297817787425, 46.766297817787425)
    elif dataset_name == "hurricane-florence":
        mean = (61.173692057426905,87.80450416631504,73.23036950130803)
        std = (41.33414392610904,39.55475091802663,43.148093428561175)
    elif dataset_name == "hurricane-harvey":
        mean = (70.16283722045026,92.65518551584839,84.64817113656672)
        std = (29.434251423775198,28.65973539773634,33.78763387568076)
    elif dataset_name == "hurricane-matthew":
        mean = (89.96771758156719,116.6059685849023,111.03711304029659)
        std = (34.97487680331815,31.71839030954414,37.65354432827067)
    elif dataset_name == "hurricane-michael":
        mean = (89.83525237842801,109.46362380190331,101.40280667118144)
        std = (37.66497986405848,33.99731755721387,36.54798296235526)
    elif dataset_name == "mexico-earthquake":
        mean = (80.2090893866947,99.32623525210208,105.71896962247602)
        std = (30.21822927421283,29.037648732886442,33.56884753157306)
    elif dataset_name == "midwest-flooding":
        mean = (75.49061228976089,100.71561936859513,92.07453209584153)
        std = (37.68655740482354,33.34134890278781,41.68130401613134)
    elif dataset_name == "palu-tsunami":
        mean = (97.23325185684871,109.38470481047692,111.78279427168913)
        std = (34.58485855260434,32.2007872497046,35.55297912369856)
    elif dataset_name == "santa-rosa-wildfire":
        mean = (90.67545608635851,110.3542241150215,106.92044795905343)
        std = (33.31935403792429,33.810744134644814,38.6565859235212)
    elif dataset_name == "socal-fire":
        mean = (91.18316259111282,111.95165370124616,108.30314802579456)
        std = (39.587315311348554,39.01678178641373,44.299969996194704)
    elif dataset_name == "xbd-all":
        mean = (82.55671640014648, 103.93882922363281, 98.99242678833008)
        std = (35.20874347360597, 33.589301143736115, 38.66435631993615)
    

    if dataset_name in ["cifar10", "cifar100", "eurosat", "stl10"]:
        weak_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                ## transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                ## transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        strong_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                ## RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        testing_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif dataset_name in ["abcd", "guatemala-volcano", "hurricane-florence", "hurricane-harvey", "hurricane-matthew", "hurricane-michael", "mexico-earthquake", "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire", "socal-fire", "xbd-all"]:
        weak_transforms = transforms.Compose(
            [
                NumpyHWC2TensorCHW(),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=180),
                ]),
                transforms.Normalize(mean, std),
            ]
        )
        strong_transforms = transforms.Compose(
            [
                NumpyHWC2TensorCHW(),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomOrder([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=180),
                ]),
                transforms.Normalize(mean, std),
            ]
        )
        testing_transforms = transforms.Compose([NumpyHWC2TensorCHW(), transforms.Normalize(mean, std)])
    else:
        raise NotImplementedError("Not implemented.")

    return weak_transforms, strong_transforms, testing_transforms