import math

import torch
import torch.nn as nn

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # box정보가 (center_x, center_y, w, h)형태라면 (lt_x, lt_y, rb_x, rb_y)형태로 변환
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else: 
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection 구하기
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area 구하기
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU: # GIoU, DIoU, CIoU란? : https://silhyeonha-git.tistory.com/3
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def compute_loss(predictions, targets, model):

    device = targets.device

    # 세 가지 loss에 대한 placeholder를 각각 지정     # placeholder : loss의 누적합을 어떤 변수에 담고 싶을 때 해당 변수의 초깃값을 0으로 설정하는 것
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # 예측값의 Loss를 계산하기 위해 ground truth 형성
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # cls_score와 obj_score에 대한 loss function을 다르게 적용
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # 3개의 YOLOLayer(13, 26, 52)에 대해 각각 Loss 계산
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # num_targets=1 이상일 때 loss를 계산한다. (0은 False로 취급되어 if문을 통과하지 못한다.)
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, torch.cat((lbox, lobj, lcls, loss)).detach().cpu()


def build_targets(p, targets, model):
    '''
    type(p) : list
    len(p) : 3
        p[0].shape : (bs, 3, 13, 13, 85)
        p[1].shape : (bs, 3, 26, 26, 85)
        p[2].shape : (bs, 3, 52, 52, 85)
    targets example:   
        45 0.479492 0.688771 0.955609 0.595500 
        45 0.736516 0.247188 0.498875 0.476417 
        50 0.637063 0.732938 0.494125 0.510583 
        45 0.339438 0.418896 0.678875 0.781500 
        49 0.646836 0.132552 0.118047 0.096937 
        49 0.773148 0.129802 0.090734 0.097229 
        49 0.668297 0.226906 0.131281 0.146896 
        49 0.642859 0.079219 0.148063 0.148062 
    targets.shape : (n_object, 5) # n_object : 한 이미지 내에 존재하는 객체의 수,  5 : class, x, y, w, h 

    '''

    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # na : 하나의 grid cell에 할당된 앵커박스의 개수
    # nt : 한 이미지 내에 존재하는 객체의 수
    na, nt = 3, targets.shape[0] #TODO
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    
    # ai, ai[:, :, None] : colab의 YOLOv3_shape_test.ipynb 참조
    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    '''
    tragets example:
        45 0.479492 0.688771 0.955609 0.595500 0
        45 0.736516 0.247188 0.498875 0.476417 0
        50 0.637063 0.732938 0.494125 0.510583 0
        45 0.339438 0.418896 0.678875 0.781500 0
        49 0.646836 0.132552 0.118047 0.096937 0
        49 0.773148 0.129802 0.090734 0.097229 0
        49 0.668297 0.226906 0.131281 0.146896 0
        49 0.642859 0.079219 0.148063 0.148062 0
        
        45 0.479492 0.688771 0.955609 0.595500 1
        45 0.736516 0.247188 0.498875 0.476417 1
        50 0.637063 0.732938 0.494125 0.510583 1
        45 0.339438 0.418896 0.678875 0.781500 1
        49 0.646836 0.132552 0.118047 0.096937 1
        49 0.773148 0.129802 0.090734 0.097229 1
        49 0.668297 0.226906 0.131281 0.146896 1
        49 0.642859 0.079219 0.148063 0.148062 1

        45 0.479492 0.688771 0.955609 0.595500 2
        45 0.736516 0.247188 0.498875 0.476417 2
        50 0.637063 0.732938 0.494125 0.510583 2
        45 0.339438 0.418896 0.678875 0.781500 2
        49 0.646836 0.132552 0.118047 0.096937 2
        49 0.773148 0.129802 0.090734 0.097229 2
        49 0.668297 0.226906 0.131281 0.146896 2
        49 0.642859 0.079219 0.148063 0.148062 2
    '''

    for i, yolo_layer in enumerate(model.yolo_layers):
        '''
        yolo_layer.anchors : 각 yolo_layer에서 사용하는 앵커박스의 크기 
            13*13 : [(116,90),  (156,198),  (373,326)]
            26*26 : [(30,61),  (62,45),  (59,119)]
            52*52 : [(10,13),  (16,30),  (33,23)]
        yolo_layer.stride : 각 yolo_layer를 통과했던 이미지의 크기를 13 or 26 or 52로 나눈 몫
        '''
        # 각 앵커박스의 크기는 원본 이미지 기준이므로 이를 stride로 나누어 grid의 resolution(13 or 26 or 52)에 scaling수행
        # (yolo_layer.stride는 원본이미지의 크기가 13 or 26 or 52보다 몇 배정도 더 큰지를 담고 있음. 예를들어 원본 416에 13*13 grid라면 416//13=32, 즉 stride=32가 된다 )
        
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        
        anchors = yolo_layer.anchors / yolo_layer.stride
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
          # gain[2:6]은 x, y, w, h에 대응한다.
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # p[i].shape : (bs, 3, 52, 52, 85)라면 torch.tensor(p[i].shape)[[3, 2, 3, 2]] == tensor([52, 52, 52, 52])
        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        t = targets * gain
        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j]
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long()
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long()
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # Add correct anchor for each target to the list
        anch.append(anchors[a])
        # Add class for each target to the list
        tcls.append(c)

    return tcls, tbox, indices, anch
