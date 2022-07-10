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
        # c박스란? : 최소한의 면적으로 gt와 pred를 모두 감싸는 박스
        # GIoU는 c박스의 면적을 활용하고 DIoU와 CIoU는 c박스의 대각선 길이를 활용한다.
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # (c박스의 대각선 길이)^2
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # (gt와 pred간의 center좌표 거리)^2
            if DIoU:
                return iou - rho2 / c2  # Loss_DIoU : 1 - iou + (rho2 / c2)
            elif CIoU:  
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # Loss_CIoU : 1 - iou + (rho2 / c2 + v * alpha)
        else:  # GIoU 
            c_area = cw * ch + eps  # c박스의 면적
            return iou - (c_area - union) / c_area  # Loss_GIoU : 1 - iou + (c_area - union) / c_area
    else:
        return iou  # IoU


def compute_loss(predictions, targets, model):

    device = targets.device

    # 세 가지 loss에 대한 placeholder를 각각 지정     # placeholder : loss의 누적합을 어떤 변수에 담고 싶을 때 해당 변수의 초깃값을 0으로 설정하는 것
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)


    # 예측값의 Loss를 계산하기 위해 gt 정보 형성
    # 각 target(하나의 gt_box)에 대하여 다음의 정보를 담고있음. 
    # (class), (x, y, w, y), (img_id, 앵커 id, grid id_y, grid id_x), (앵커박스 초기 w, h)
    
    # tcls : [13_class, 26_class, 52_class]
    # tbox : [13_(x, y, w, h), 26_(x, y, w, h), 52_(x, y, w, h)]
    # indices : [ // ]
    # anchors : [ // ]
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # cls_score와 obj_score에 대해 각각 loss function 적용
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # 3개의 YOLOLayer(13, 26, 52)에 대해 각각 Loss 계산
    for layer_index, layer_predictions in enumerate(predictions):
        # (img_id, 앵커 id(0~2), grid id_y, grid id_x)
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

            # box Regression
            # 박스 중심좌표인 x, y에 대한 예측이 grid cell 밖에 형성되는 것을 막기위해 sigmoid로 scale
            pxy = ps[:, :2].sigmoid()
            
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            # 하나의 grid cell에는 세 개의 앵커박스가 존재하는데, target의 크기를 고려하여 적합한 앵커박스를 선택할 필요가 있다.
            # (꼭 하나만 선택하는건 아니고 세 개 다 target에 적합하다면 세 개 다 선택한다.)
            # build_targets 함수에서 target 좌표를 바탕으로 각 grid_size(13, 26, 52)별로 적합한 앵커박스를 선별했었다. (r < 4)
            # 선별한 앵커박스를 line 94의 인덱싱을 통해 변수 ps에 선언 후 각자의 크기에 맞춰 pwh를 scale해주었다.
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            
            pbox = torch.cat((pxy, pwh), 1)
            
            # build_targets에서 산출한 모든 앵커박스와 target간의 CIoU를 구한다.
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # iou loss
            lbox += (1.0 - iou).mean()  

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
        (# 이 예시는 batch_size=1, 다시 말해 targets에 하나의 이미지에 대한 target 정보만 들어올 경우를 가정하여 작성함. 따라서 예시의 img_id의 값은 모두 같다.)
        (# batch_size=32일 경우 차원은 달라지지 않지만 torch.tensor.cat([img1, img2, img3, ..., img32], axis=1)의 형태로 concatenate되어 표현되므로 행의 개수가 매우 많아지게 된다.)
        img_id 45 0.479492 0.688771 0.955609 0.595500 
        img_id 45 0.736516 0.247188 0.498875 0.476417 
        img_id 50 0.637063 0.732938 0.494125 0.510583 
        img_id 45 0.339438 0.418896 0.678875 0.781500 
        img_id 49 0.646836 0.132552 0.118047 0.096937 
        img_id 49 0.773148 0.129802 0.090734 0.097229 
        img_id 49 0.668297 0.226906 0.131281 0.146896 
        img_id 49 0.642859 0.079219 0.148063 0.148062
    # img_id : 한 batch내에서의 id이다. batch_size=32라면 0~31 사이의 값이 된다.
    
    
    targets.shape : (n_object, 6) 
    # n_object : 한 이미지 내에 존재하는 객체의 수
    # 6 : img_id, class, x, y, w, h 
    '''

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
        img_id 45 0.479492 0.688771 0.955609 0.595500 0
        img_id 45 0.736516 0.247188 0.498875 0.476417 0
        img_id 50 0.637063 0.732938 0.494125 0.510583 0
        img_id 45 0.339438 0.418896 0.678875 0.781500 0
        img_id 49 0.646836 0.132552 0.118047 0.096937 0
        img_id 49 0.773148 0.129802 0.090734 0.097229 0
        img_id 49 0.668297 0.226906 0.131281 0.146896 0
        img_id 49 0.642859 0.079219 0.148063 0.148062 0
 
        img_id 45 0.479492 0.688771 0.955609 0.595500 1
        img_id 45 0.736516 0.247188 0.498875 0.476417 1
        img_id 50 0.637063 0.732938 0.494125 0.510583 1
        img_id 45 0.339438 0.418896 0.678875 0.781500 1
        img_id 49 0.646836 0.132552 0.118047 0.096937 1
        img_id 49 0.773148 0.129802 0.090734 0.097229 1
        img_id 49 0.668297 0.226906 0.131281 0.146896 1
        img_id 49 0.642859 0.079219 0.148063 0.148062 1

        img_id 45 0.479492 0.688771 0.955609 0.595500 2
        img_id 45 0.736516 0.247188 0.498875 0.476417 2
        img_id 50 0.637063 0.732938 0.494125 0.510583 2
        img_id 45 0.339438 0.418896 0.678875 0.781500 2
        img_id 49 0.646836 0.132552 0.118047 0.096937 2
        img_id 49 0.773148 0.129802 0.090734 0.097229 2
        img_id 49 0.668297 0.226906 0.131281 0.146896 2
        img_id 49 0.642859 0.079219 0.148063 0.148062 2
        
    targets[:, :, 5].unique() = tensor([0, 1, 2]) 인 이유 
    하나의 Grid cell당 세 개의 predicted 앵커박스가 존재하는데, 
    박스 하나당 개별적으로 target 데이터를 할당하기 위해 똑같은 target 데이터 세 개를 concatenate후 0, 1, 2의 인덱스를 부여시킴     
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
        # 박스를 Grid cell 개수만큼 target의 좌표데이터를 scale해줌
        t = targets * gain
        # Check if we have targets
        if nt:
            '''
            t : tensor([[[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  0.0000],
                         [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  0.0000],
                         [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  0.0000],
                         [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  0.0000],
                         [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  0.0000],
                         [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  0.0000],
                         [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  0.0000],
                         [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  0.0000]],
 
                        [[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  1.0000],
                         [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  1.0000],
                         [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  1.0000],
                         [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  1.0000],
                         [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  1.0000],
                         [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  1.0000],
                         [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  1.0000],
                         [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  1.0000]],
 
                        [[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  2.0000],
                         [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  2.0000],
                         [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  2.0000],
                         [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  2.0000],
                         [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  2.0000],
                         [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  2.0000],
                         [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  2.0000],
                         [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  2.0000]]])

            anchors[:, None] : tensor([[[10, 13]],

                                       [[16, 30]],

                                       [[33, 23]]])
                              
            r : tensor([[[0.0082, 0.0066],
                         [0.0043, 0.0053],
                         [0.0043, 0.0057],
                         [0.0059, 0.0087],
                         [0.0010, 0.0011],
                         [0.0008, 0.0011],
                         [0.0011, 0.0016],
                         [0.0013, 0.0016]],
 
                        [[0.0061, 0.0030],
                         [0.0032, 0.0024],
                         [0.0032, 0.0026],
                         [0.0044, 0.0039],
                         [0.0008, 0.0005],
                         [0.0006, 0.0005],
                         [0.0008, 0.0007],
                         [0.0009, 0.0007]],
 
                        [[0.0026, 0.0018],
                         [0.0013, 0.0015],
                         [0.0013, 0.0016],
                         [0.0018, 0.0024],
                         [0.0003, 0.0003],
                         [0.0002, 0.0003],
                         [0.0004, 0.0005],
                         [0.0004, 0.0005]]])
                         
            j : tensor([[False,  True,  True,  True,  True,  True,  True,  True],
                        [ True,  True,  True,  True, False, False,  True,  True],
                        [ True,  True,  True,  True, False, False, False, False]])
            '''
            # r(ration) : (target의 w, h) / (앵커박스의 초기 w, h크기)
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # max(r, 1. / r) : 두 숫자가 서로 n배(n>=1) 차이난다는 insight를 얻기위해 나눗셈을 실행한다고 가정해보자. 
            #                  8/2 의 경우 r = 4 이므로 바로 insight로 활용할 수 있지만 
            #                  2/8 의 경우 r = 0.25 이므로 역수를 취해줘야 insight로 활용할 수 있다.
            # .max(2)[0] : w, h중 r값이 큰 쪽만 판단 기준으로서 사용하겠다는 의미이다. 
            #              [0]으로 인덱싱하는 이유는 torch.tensor.max는 (max_values, max_indices)의 형태의 반환값을 가지기 때문이다.
            # < 4 : r이 4미만인 것만 사용하겠다는 의미이다.
            #       target에 비해 앵커박스의 크기가 너무 크거나 작을 경우 regression이 어려워지므로 그 기준을 r < 4로 정한 것이다. 
            #       쉬운 예를 들자면 anchors가 52*52 grid의 앵커박스 정보를 담고있다 가정했을 때,
            #       r < 4의 기준에 부합하지 못하는 target은 26*26이나 13*13 grid의 앵커박스가 담당하게 된다. 

            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # 각 앵커의 id(0 ~ 2)는 t[:, :, 6]에 저장되어 있다.
            t = t[j]
            '''
            [[[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  0.0000],
              [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  0.0000],
              [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  0.0000],
              [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  0.0000],
              [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  0.0000],
              [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  0.0000],
              [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  0.0000],
              [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  0.0000]],
  
             [[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  1.0000],
              [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  1.0000],
              [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  1.0000],
              [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  1.0000],
              [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  1.0000],
              [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  1.0000],
              [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  1.0000],
              [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  1.0000]],
  
             [[ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  2.0000],
              [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  2.0000],
              [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  2.0000],
              [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  2.0000],
              [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  2.0000],
              [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  2.0000],
              [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  2.0000],
              [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  2.0000]]]
            위와 같은 3차원 텐서였던 t가
            
            [[ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  0.0000],
             [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  0.0000],
             [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  0.0000],
             [ img_id, 49.0000, 33.6355,  6.8927,  6.1384,  5.0407,  0.0000],
             [ img_id, 49.0000, 40.2037,  6.7497,  4.7182,  5.0559,  0.0000],
             [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  0.0000],
             [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  0.0000],
             [ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  1.0000],
             [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  1.0000],
             [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  1.0000],
             [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  1.0000],
             [ img_id, 49.0000, 34.7514, 11.7991,  6.8266,  7.6386,  1.0000],
             [ img_id, 49.0000, 33.4287,  4.1194,  7.6993,  7.6992,  1.0000],
             [ img_id, 45.0000, 24.9336, 35.8161, 49.6917, 30.9660,  2.0000],
             [ img_id, 45.0000, 38.2988, 12.8538, 25.9415, 24.7737,  2.0000],
             [ img_id, 50.0000, 33.1273, 38.1128, 25.6945, 26.5503,  2.0000],
             [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  2.0000]]
            t = t[j]를 통해 위와 같은 2차원 텐서로 바뀌었다. 
            j에서 False가 7개 검출됨에 따라 3*8 = 24개였던 박스가 24 - 7 = 17개로 줄어들은 것을 볼 수 있다. 

            '''
        else:
            t = targets[0]

        # b : img_id (b는 batch내에서의 index라는 뜻)
        # c : class_id (위에서 사용한 예시대로라면 tensor([45, 50, 45, 49, 49, 49, 49, 45, 45, 50, 45, 49, 49, 45, 45, 50, 45])가 된다.)
        b, c = t[:, :2].long().T
 
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        # 현재 x, y, w, h는 grid에 맞춰 scale된 상태이다. 예를들어 x = 1.2라면 grid cell 하나의 너비의 1.2배만큼의 값을 가진다.
        # 이 때 grid cell 하나의 너비는 학습 과정에서 Loss계산할 땐 1로 두고, 추론결과를 최종 이미지로 산출할 땐 stride를 곱하여 원본 이미지 크기에 맞춰 scale된다.)
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        # gij : cell index를 얻기 위해 gxy를 int로 변환한 값. cell index는 models.py의 YOLOLayer의 _make_grid메소드를 통해 이미 생성된 상태.
        gij = gxy.long()
        gi, gj = gij.T

        # a : 앵커 id(0 ~ 2)를 int로 변환한 값
        a = t[:, 6].long()
        
        # 각 target의 (img_id, 앵커 id, grid id_y, grid id_x) 획득
        # clamp_(min, max) : out of bounds오류를 방지하기 위해 index값을 제한. 52*52 grid라면 clamp_(0, 51)이 됨
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        
        # 각 target의 (x, y, w, h) 획득
        # grid 전체를 기준으로 한 global x, y좌표를 cell 하나를 기준으로 한 local 좌표로 변환함
        #   예를 들어 global x, y좌표가 (38.2988, 12.8538)라면 local 좌표는 (0.2988, 0.8538)가 된다.
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        
        # 각 target에 할당된 앵커박스의 초기 w, h 획득
        #   예를 들어 target이 t[16] = [ img_id, 45.0000, 17.6508, 21.7826, 35.3015, 40.6380,  2.0000]라면 
        #   앵커 id는 t[:, :, 6]에 저장되어 있으므로 [(10, 13), (16, 30), (33, 23)]의 index2에 접근해 (33,23)를 획득한다.
        anch.append(anchors[a])
        
        # 각 target의 class를 획득
        tcls.append(c) 

    # (class), (x, y, w, h), (img_id, 앵커 id, grid id_y, grid id_x), (앵커박스 초기 w, h)
    return tcls, tbox, indices, anch
