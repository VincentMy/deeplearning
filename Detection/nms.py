import numpy as np
def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #print("areas:",areas.shape)#areas: (3137,)
    #print("end:",end)
    #argsort()返回的是数组值从小到大的索引值
    #argsort()[::-1]返回的是数组从大到小的索引值
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        #返回最大值的索引值
        i = order[0]
        keep.append(i)
        #计算橡胶区域左上以及右下的坐标
        #其中xx1,yy1,xx2,yy2表示的是数组
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        order = order[inds + 1]

    return keep
