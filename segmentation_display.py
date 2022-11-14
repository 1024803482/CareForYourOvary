import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2


def create_pascal_label_colormap(map):
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap

    返回:
        可视化分割结果的颜色映射Colormap
    """
    # colormap = np.zeros((256, 3), dtype=int)
    # ind = np.arange(256, dtype=int)

    # for shift in reversed(range(8)):
    #    for channel in range(3):
    #        colormap[:, channel] |= ((ind >> channel) & 1) << shift
    #    ind >>= 3
    colormap = np.array([[0, 0, 0],
                         map, ])
    return colormap


def label_to_color_image(label, map):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap(map)

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation2(image, seg_map, map):
    """
    输入图片和分割 mask 的统一可视化.
    """
    seg_image = label_to_color_image(seg_map, map).astype(np.uint8)
    output = np.uint8(0.7 * image + 0.3 * seg_image)
    return output