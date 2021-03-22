# -*- coding: utf-8 -*-
"""
@Time   : 2021/3/22 上午11:35
@Author : Li Shenzhen
@File   : soft_nms.py
@Software:PyCharm
soft nms 实现代码
"""

from torch import Tensor
import torch


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  # 删除面积小于0 不相交的  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  # 切片的用法 相乘维度减1
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def soft_nms(boxes: Tensor, scores: Tensor, soft_threshold=0.01,  iou_threshold=0.7, weight_method=2, sigma=0.5):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（选取的得分TopK）之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 （手动设置）
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
     # 值从小到大的 索引， 索引对应的 是 元boxs索引 scores索引
    idxs = scores.argsort()
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort()  # 评分排序

        if idxs.size(0) == 1:  # 就剩余一个框了；
            keep.append(idxs[-1])  # 位置不能边
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]  # [?, 4]
        keep.append(max_score_index)  # 位置不能边
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        if weight_method == 1:   # 线性抑制  # 整个过程 只修改分数
            ge_threshod_bool = ious[0] >= iou_threshold
            ge_threshod_idxs = idxs[ge_threshod_bool]
            scores[ge_threshod_idxs] *= (1. - ious[0][ge_threshod_bool])  # 小于IoU阈值的不变
            # idxs = idxs[scores[idxs] >= soft_threshold]  # 小于soft_threshold删除， 经过抑制后 阈值会越来越小；
        elif weight_method == 2:  # 高斯抑制， 不管大不大于阈值，都计算权重
            scores[idxs] *= torch.exp(-(ious[0] * ious[0]) / sigma) # 权重(0, 1]
            # idxs = idxs[scores[idxs] >= soft_threshold]
        # else:  # NMS
        #     idxs = idxs[ious[0] <= iou_threshold]

    # keep = scores[scores > soft_threshold].int()
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    print(keep)
    boxes = boxes[keep]  # 保留下来的框
    scores = scores[keep]  # soft nms抑制后得分
    return boxes, scores

def test2():
    pro_boxes = torch.load("Boxes_forNMS_test.pt")
    one_boxes = pro_boxes["boxes"][0]  # [1000, 4]
    one_scores = pro_boxes["scores"][0]  # [1000]
    boxes, scores = soft_nms(one_boxes, one_scores, soft_threshold=0.0010, iou_threshold=0.7, weight_method=2, sigma=0.5)
    print(scores.shape)

def test1():
    boxes = torch.tensor([[200, 200, 400, 400],
                          [220, 220, 420, 420],
                          [200, 240, 400, 440],
                          [240, 200, 440, 400],
                          [1, 1, 2, 2]], dtype=torch.float)
    boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float)

    boxes, scores = soft_nms(boxes, boxscores, soft_threshold=0.001, iou_threshold=0.7, weight_method=2, sigma=0.5)
    print(boxes)
    print(scores)


if __name__ == "__main__":
    test1()
    test2()

"""
tensor([4, 0, 1, 2, 3])
tensor([[  1.,   1.,   2.,   2.],
        [200., 200., 400., 400.],
        [220., 220., 420., 420.],
        [200., 240., 400., 440.],
        [240., 200., 440., 400.]])
tensor([0.9000, 0.8000, 0.2771, 0.0977, 0.0523])
"""