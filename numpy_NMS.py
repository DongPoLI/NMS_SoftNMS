# -*- coding: utf-8 -*-
"""
@Time   : 2021/3/21 下午10:39
@Author : Li Shenzhen
@File   : numpy_NMS.py
@Software:PyCharm
"""

import numpy as np
from numpy import array


def box_area(boxes :array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1 :array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt  # 右下角 - 左上角；
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes :array, scores :array, iou_threshold :float):

    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)  # Tensor
    return keep

# 测试NMS的代码；
def test1():
    import torch
    pro_boxes = torch.load("Boxes_forNMS_test.pt")

    one_boxes = pro_boxes["boxes"][0]  # [1000, 4]
    one_scores = pro_boxes["scores"][0]  # [1000]

    keep = numpy_nms(one_boxes.numpy(), one_scores.numpy(), 0.7)
    print(keep)
    print(keep.shape)

    ## pytorch nms 接口
    import torchvision
    keep_1 = torchvision.ops.nms(one_boxes, one_scores, 0.7)
    print("#"*20)
    print(keep_1)
    print(keep_1.shape)
    print(keep_1.numpy()==keep)


if __name__ == "__main__":
    test1()