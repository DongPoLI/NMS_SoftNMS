# NMS都不会，做什么Detection！
## Non-maximum suppression（非极大值抑制）算法

### NMS原理： 
1. 首先得出所有的预测框集合`B`、 对应框的得分`Scores`， NMS(IoU)阈值`T`;
2. 定义存放侯选框的集合`H`（初始为`Null`）， 对`Scores`排序选出得分最大的框为`maxBox`， 
    将`maxBox`从集合`B`中移到集合H中，集合`B`中没有`maxBox`框了;
3. 计算`maxBox`和`B`中剩余的所有框的IoU, 将IoU大于`T`的从`B`中删除(认为和`maxBox`重叠了);
4. 重复2～3步骤，直到集合`B`为`Null`， 集合H中存放的框就是NMS处理的结果；
    
    重复步骤是：  
    （1）对集合B中剩余框对应的得分进行排序， 选出最大得分的框maxBox，并从集合B中移到集合H中。
    
    （2） 计算这个得分最大的框maxBox和集合B中框的IoU阈值，将大于IoU阈值的框从B中删除。
    
### NMS代码实现
#### 1. Pytorch代码实现 
```python
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

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  

    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（NMS之前选取过得分TopK）之后， 在传入之前处理好的；
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    idxs = scores.argsort()  # 值从小到大的 索引

    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = idxs.new(keep)  # Tensor
    return keep
```

#### 2. Pytorch代码实现
```python
import torch
def nms(boxes, scores, overlap=0.7, top_k=200):
    """
    输入:
        boxes: 存储一个图片的所有预测框。[num_positive,4].
        scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
        overlap: nms抑制时iou的阈值.
        top_k: 先选取置信度前top_k个框再进行nms.
    返回:
        nms后剩余预测框的索引.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    # 保存留下来的box的索引 [num_positive]
    # 函数new(): 构建一个有相同数据类型的tensor

    # 如果输入box为空则返回空Tensor
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]  # x1 坐标
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
    v, idx = scores.sort(0)  # 升序排序
    idx = idx[-top_k:]  # 前top-k的索引，从小到大
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()  # new() 无参数，创建 相同类型的空值；
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # 目前最大score对应的索引  # 选取得分最大的框索引；
        keep[count] = i  # 存储在keep中
        count += 1
        if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
            break
        idx = idx[:-1]  # 去掉最后一个

        # 剩下boxes的信息存储在xx，yy中
        torch.index_select(x1, 0, idx, out=xx1)  # 从x1中再维度0选取索引为idx 数据 输出到xx1中；
        torch.index_select(y1, 0, idx, out=yy1)  # torch.index_select（） # 从tensor中按指定维度和索引 取值；
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # 计算当前最大置信框与其他剩余框的交集，不知道clamp的同学确实容易被误导
        xx1 = torch.clamp(xx1, min=x1[i])  # max(x1,xx1)  # x1 y1 的最大值
        yy1 = torch.clamp(yy1, min=y1[i])  # max(y1,yy1)
        xx2 = torch.clamp(xx2, max=x2[i])  # min(x2,xx2)  # x2 x3 最小值；
        yy2 = torch.clamp(yy2, max=y2[i])  # min(y2,yy2)
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
        h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
        w = torch.clamp(w, min=0.0)  # max(w,0)
        h = torch.clamp(h, min=0.0)  # max(h,0)
        inter = w * h

        # 计算当前最大置信框与其他剩余框的IOU
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # 剩余的框的面积
        union = rem_areas + area[i] - inter  # 并集
        IoU = inter / union  # 计算iou

        # 选出IoU <= overlap的boxes(注意le函数的使用)
        idx = idx[IoU.le(overlap)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
    return keep, count
```
参考自：[链接](https://blog.csdn.net/sinat_37145472/article/details/90232622)

#### 3. Numpy代码实现
```python
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
    wh = rb - lt
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
```



