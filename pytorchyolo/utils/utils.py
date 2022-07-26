import time
import platform
import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import subprocess
import random
import imgaug as ia


def load_classes(path):
    """
    path : [workspace]/data/coco.names
    coco데이터셋의 label 목록을 불러옴 (person, bicycle, car, motorbike 등 총 80개)
    
    """
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names
 
 
def weights_init_normal(m):
    '''
    가중치 초기화
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

    # xywh2xyxy
    # input : (center_x, center_y, w, h)  ==>  b_box의 중심좌표(x, y)와 b_box의 가로세로인 (w, h)로 구성
    # output : (lt_x, lt_y, rb_x, rb_y)  ==>  좌상단 좌표와 우하단 좌표로 구성
    
    # test.py에서 실제 b_box와 예측 b_box간의 IOU를 구하기 위해 사용
def xywh2xyxy(x): 
    y = x.new(x.shape) # torch관련 함수같은데 검색해도 마땅히 나오는게 없음
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

    # transforms.py에서 이미지와 b_box를 동시에 augmentaiton할 때 사용
def xywh2xyxy_np(x):
    y = np.zeros_like(x) # np.zero와의 차이 : np.zero는 shape를 인자로 받지만 np.zeros_like는 np.array 그 자체를 받아서 해당 array와 같은 shape의 영행렬을 반환함
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

def soft_nms_pytorch(dets, box_scores, sigma=0.6, thresh=0.4, cuda=0):
    """
    # Augments
        dets:        박스 좌표를 나타내는 2차원 tensor ([[y1, x1, y2, x2],[y1, x1, y2, x2],....])
        box_scores:  각 box의 confidence score (obj conf * cls conf)
        sigma:       높을수록 suppression기준이 완화됨 
        thresh:      낮을수록 suppression기준이 완화됨
        cuda:        1이면 cuda, 0이면 cpu
    # Return
        최종 선택된 b_box들의 인덱스를 반환
    """

    # 각 box에 index를 매긴 후 axis=1의 맨 뒤에 concatenate
    # b_box들의 index가 nms과정을 거친 후 이리저리 바뀌기 때문에 nms적용 전의 초기 index를 b_box좌표와 함께 저장해두기 위함
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # 면적 계산을 위해 y1, x1, y2, x2 각각의 값을 변수로 선언
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        #  max_score가 t_score보다 큰 경우 (= 현재(i)박스보다 다른 박스의 conf가 더 높은 경우) 위치를 서로 바꿈
        t_score = scores[i].clone()
        other = i + 1

        if i != N - 1: # i == N-1이면 max_other가 존재하지 않으므로 조건문이 필요
            max_score, max_other = torch.max(scores[other:], dim=0)
            if t_score < max_score: 
                dets[i], dets[max_other.item() + i + 1] = dets[max_other.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[max_other.item() + i + 1] = scores[max_other.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[max_other + i + 1] = areas[max_other + i + 1].clone(), areas[i].clone()

        # IoU를 계산하기 위해 intersection의 좌표 (yy1, xx1, yy2, xx2)를 구함
        xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[other:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[other:, 1].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[other:, 2].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[other:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, yy2 - yy1 + 1)
        h = np.maximum(0.0, xx2 - xx1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h) # intersection의 넓이
        ovr = torch.div(inter, (areas[i] + areas[other:] - inter)) # IOU(intersection over union)

    #     # exponential decay를 통해 각 b_box score에 가중치를 곱해주는 과정 (시그마가 클 수록 weight가 커짐 --> suppression기준이 완화됨)
    #     weight = torch.exp(-(ovr * ovr) / sigma) # weight < 1, max_score b_box와의 IOU(=ovr)가 높을 수록 exponential 값이 낮아짐
    #     scores[other:] = weight * scores[other:] # IOU가 일정 이상이면 아예 삭제해버리는 초기 nms와 다르게 weight를 곱해줘서 연속적인 scoring을 수행 

    # # nms결과가 thresh보다 큰 박스의 인덱스를 keep로 선언
    # # 초기 nms의 nms_thres와 여기서 사용한 thresh의 기준이 달라서 헷갈리지 않도록 유의 (전자는 높을수록 suppression기준이 완화되고 후자는 낮을수록 suppression기준이 완화됨)
    # sigma : 높을수록 suppression 기준이 완화됨
    # keep = dets[:, 4][scores > thresh].int()
        # exponential decay
        weight = torch.exp(-(ovr * ovr) / sigma) # weight < 1 
        scores[other:] = weight * scores[other:] 

    # nms로 최종 선택된 b_box의 초기 인덱스를 keep로 선언
    keep = dets[:, 4][scores > thresh].long()

    return keep


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, soft_nms=False, sigma=0.6, thresh=0.4):
    """
    예측된 b_box들에 대하여 NMS 수행
    prediction.shape : (bs, 8112, 85)
    conf_thres : class n에 대한 confidence score (ground truth에 대한 IOU * class에 대한 softmax score)
    nms_thres
    
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # 80

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # 한 이미지당 최대 검출 개수 (이 이상이 검출되면 무시하고 300개를 유지함)
    max_nms = 30000  # nms에 넣을 최대 box 개수 (max_nms = 30000이라면 obj conf가 높은 순대로 30000개만 슬라이싱) (maximum number of boxes into torchvision.ops.nms())
    time_limit = 1.0  # nms 최대 수행 시간 (nms process에 time_limit이상의 시간이 소요되면 for문을 탈출)
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    '''
    이미지 하나당 box의 최대 개수는 10647개인데 어째서 max_nms=30000이라는 수치가 가능한가?

    [x1, y1, x2, y2, obj_conf, conf(class14), conf(class31), conf(class59)]
    위와 같은 형태를

    [[x1, y1, x2, y2, conf, 14],
     [x1, y1, x2, y2, conf, 31],
     [x1, y1, x2, y2, conf, 59]]
    이렇게 바꿈으로써 box 좌표 한 개에 대한 정보가 conf_thres보다 높은 클래스 개수 만큼 중복되어 존재하게 되어 
    이미지 내의 객체가 많을 경우 30000개를 가볍게 넘는 경우가 생길 수 있다.
    여담으로 이렇게 class 하나당 box하나를 차지하게 됨으로서 클래스별로 AP를 계산하여 mAP를 얻거나 지정한 class만을 따로 검출하는 것이 가능해진다.
    '''

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    # prediction의 shape : (bs, 10647, 85) (bs: 배치사이즈, 10647(=3*13^2 + 3*26^2 + 3*52^2: 예측된 박스 개수)
    # 각각의 이미지에 대해 nms를 수행하기 위해 for문을 돌린다. bs = 32라면 반복 횟수가 32번이다.
    for xi, x in enumerate(prediction):
        '''
        x[..., 4] : 예측된 b_box들의 objectness confidence
        x[..., 5] : 예측된 b_box들의 class0 confidence
        ~~~~ 
        x[..., 84] : 예측된 b_box들의 class79 confidence
        '''
        x = x[x[..., 4] > conf_thres] # objectness confidence가 conf_thres이상인 x만 추출
        
        # x.shape[0]가 0일 경우 (= obj conf가 conf_thres이상인 box가 없을 경우)
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (x1, y1, x2, y2, conf, cls)
        if multi_label: # class가 두 개 이상이면 
            # conf가 conf_thres보다 큰 인덱스만 반환
            # i : conf가 conf_thres보다 큰 box의 인덱스
            # j : 각 box 별 conf_thres보다 큰 conf의 인덱스 (class0이면 j=0)
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T # nonzero는 0이 아닌 원소들의 인덱스를 반환한다. 
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            '''        
            의의 : 하나의 box에 conf가 conf_thres보다 큰 클래스가 2개 이상일 경우 
            [x1, y1, x2, y2, obj_conf, conf(class14), conf(class31), conf(class59)]의 형태를
            [[x1, y1, x2, y2, conf, 14],
             [x1, y1, x2, y2, conf, 31],
             [x1, y1, x2, y2, conf, 59]]
            의 형태로 변형함. (기존에 index 4에 위치하던 obj_conf는 위에서 이미 cls_conf에 곱해서 conf를 형성함으로써 제 역할을 다했으므로 삭제함)
            
            즉 box 한 개의 좌표가 conf_thres보다 큰 conf를 가진 클래스 개수만큼 axis=1 방향으로 중복되어 표현된다.
            위의 예시에선 class14, class31, class59가 선택되었으므로 세 번 중복된다.
            
            [box0_x1, box0_y1, box0_x2, box0_y2, obj_conf, conf(class14), conf(class31), conf(class59)]
            [box1_x1, box1_y1, box1_x2, box1_y2, obj_conf, conf(class11), conf(class25), conf(class47), conf(class74), conf(class76)] 
            만약 box가 두 개 있을 때 각 box에 conf_thres를 적용한 결과가 위와 같다 가정하면
            
            [[box0_x1, box0_y1, box0_x2, box0_y2, conf, 14],
             [box0_x1, box0_y1, box0_x2, box0_y2, conf, 31],
             [box0_x1, box0_y1, box0_x2, box0_y2, conf, 59],
             [box1_x1, box1_y1, box1_x2, box1_y2, conf, 11],
             [box1_x1, box1_y1, box1_x2, box1_y2, conf, 25],
             [box1_x1, box1_y1, box1_x2, box1_y2, conf, 47],
             [box1_x1, box1_y1, box1_x2, box1_y2, conf, 74],
             [box1_x1, box1_y1, box1_x2, box1_y2, conf, 76]]      
            이렇게 각각 세 번, 다섯 번씩 중복된다.
            '''
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # conf를 내림차순으로 정렬 후 max_nms만큼만 뽑아내고 나머지는 버림
            x = x[x[:, 4].argsort(descending=True)[:max_nms]] 
        
        if soft_nms:
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = soft_nms_pytorch(boxes, scores, sigma=sigma, thresh=thresh, cuda=1)
        else:
            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            '''
            c를 각 box의 모든 좌표에 더해주는 이유
            다양한 class의 box가 들어왔을때 이를 구분하기 위해 c를 더해 다른 class간 box들의 좌표계를 다르게 해서 torchvision.nms()함수가 다른 class의 box로 인식해서 살려두게 하기 위함이다.
            ( = torchvision.ops.nms()가 box의 class를 구분하지 못하기 때문)
            '''
            # boxes (offset by class)
            # scores : obj conf
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
