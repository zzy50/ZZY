import time
import numpy as np
import torch

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    # Augments
        dets:        박스 좌표를 나타내는 2차원 tensor ([[y1, x1, y2, x2],[y1, x1, y2, x2],....])
        box_scores:  각 box의 confidence score (obj conf * cls conf)
        sigma:       gaussian decay를 얼마나 너그럽게 줄지 설정 
        thresh:      낮을수록 너그러움
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
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
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
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[other:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[other:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[other:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[other:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h) # intersection의 넓이
        ovr = torch.div(inter, (areas[i] + areas[other:] - inter)) # IOU(intersection over union)

    #     # exponential decay를 통해 각 b_box score에 가중치를 곱해주는 과정 (시그마가 클 수록 weight가 커짐 --> nms과정이 조금 더 너그러워짐)
    #     weight = torch.exp(-(ovr * ovr) / sigma) # weight < 1, max_score b_box와의 IOU(=ovr)가 높을 수록 exponential 값이 낮아짐
    #     scores[other:] = weight * scores[other:] # IOU가 일정 이상이면 아예 삭제해버리는 초기 nms와 다르게 weight를 곱해줘서 연속적인 scoring을 수행 

    # # nms결과가 thresh보다 큰 박스의 인덱스를 keep로 선언
    # # 초기 nms의 nms_thres와 여기서 사용한 thresh의 기준이 달라서 헷갈리지 않도록 유의 (전자는 높을수록 너그럽고 후자는 낮을수록 너그럽다)
    # keep = dets[:, 4][scores > thresh].int()
        # exponential decay
        weight = torch.exp(-(ovr * ovr) / sigma) # weight < 1 
        scores[other:] = weight * scores[other:] 

    # nms로 최종 선택된 b_box의 초기 인덱스를 keep로 선언
    keep = dets[:, 4][scores > thresh].int()

    return keep


def speed():
    boxes = 1000*torch.rand((1000, 100, 4), dtype=torch.float)
    boxscores = torch.rand((1000, 100), dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(1000):
        soft_nms_pytorch(boxes[i], boxscores[i], cuda=cuda)
    end = time.time()
    print("Average run time: %f ms" % (end-start))
