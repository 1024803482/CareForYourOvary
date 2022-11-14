import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio

import recognition_models


class GradCAM:
    def __init__(self, model: nn.Module, name="ResNeXt50", size=(224, 224), num_cls=1000, mean=None, std=None) -> None:
        self.model = model
        self.model.eval()

        # register hook
        # 可以自己指定层名，没必要一定通过target_layer传递参数
        # self.model.layer4
        if name in ['ResNeXt50', "ResNet50"]:
            self.model.resnet.layer4.register_forward_hook(self.__forward_hook)
            self.model.resnet.layer4.register_backward_hook(self.__backward_hook)
        elif name in ['DenseNet121', "DenseNet201"]:
            self.model.densenet.features.denseblock4.register_forward_hook(self.__forward_hook)
            self.model.densenet.features.denseblock4.register_backward_hook(self.__backward_hook)
        # getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        # getattr(self.model, target_layer).register_backward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if mean and std:
            self.mean, self.std = mean, std

        self.grads = []
        self.fmaps = []

    def forward(self, img_arr: np.ndarray, label=None):
        img_input = self.__img_preprocess(img_arr.copy())

        # forward
        output = self.model(img_input)
        output = output.cpu()
        idx = np.argmax(output.data.numpy())

        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, label)

        loss.backward()

        # generate CAM
        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        # show
        cam_show = cv2.resize(cam, self.origin_size)
        img_show = img_arr.astype(np.float32) / 255
        output = self.__show_cam_on_image(img_show, cam_show )

        self.fmaps.clear()
        self.grads.clear()
        return output

    def __img_transform(self, img_arr: np.ndarray, transform: torchvision.transforms) -> torch.Tensor:
        img = img_arr.copy()  # [H, W, C]
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)  # [N,C,H,W]
        return img

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        self.origin_size = (img_in.shape[1], img_in.shape[0])  # [H, W, C]
        img = img_in.copy()
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def __compute_loss(self, logit, index=None):
        if not index:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.num_cls).scatter_(1, index, 1)
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    def __compute_cam(self, feature_map, grads):
        """
        feature_map: np.array [C, H, W]
        grads: np.array, [C, H, W]
        return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2))  # GAP
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]  # linear combination

        cam = np.maximum(cam, 0)  # relu
        cam = cv2.resize(cam, self.size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        return cam[:, :, ::-1]


if __name__ == "__main__":
    # 调用函数
    img = imageio.imread('D:/学习/研究生/实验室/AI与卵巢癌筛查/数据库搭建调研/深度学习数据集整理/0106二维超声图像分割（8类）/Image/1358.JPG')
    net = recognition_models.DenseNet121(in_channels=3,
                                         num_classes=9,
                                         pretrained=False)
    net.load_state_dict(torch.load("./weights/classification/densenet121.pth"))
    print(net)
    grad_cam = GradCAM(net, name="DenseNet121", size=(256, 256), num_cls=9,)
    cam = grad_cam.forward(img)
    plt.imshow(cam)
    plt.show()
