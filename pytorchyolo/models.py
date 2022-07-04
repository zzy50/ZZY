from itertools import chain
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorchyolo.utils.parse_config import parse_model_config
from pytorchyolo.utils.utils import weights_init_normal


def create_modules(module_defs):
    """
    yolov3.cfg를 사용하여 모델을 구축
    module_defs : parse_model_config함수의 output
    
    """

    hyperparams = module_defs.pop(0) # yolov3.cfg의 [net]의 속성을 가져옴 (pop으로 가져왔으므로 module_defs에 있는 [net]요소가 삭제됨) 
    # 파싱한 하이퍼파라미터 값이 전부 문자열이므로 정수형 또는 실수형으로 변경 
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'], # yolov3.cfg에는 policy=step으로 되어있는데, lr_scheduler의 타입을 명시한 것 같음. train.py에 직접 스케쥴러를 정의했으므로 딱히 사용되는 곳은 없다  
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs): # 한 번 돌때마다 레이어 하나씩 생성
        modules = nn.Sequential()
        
        if module_def["type"] == "convolutional":

            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn, # not bn = not 0 = True
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky": # yolov3에선 shortcut에 linear를 사용하고 그 외에는 전부 leaky를 사용한다
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "mish": # yolov3에선 안씀
                modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool": # yolov3에선 안씀
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample": # F.interpolate로 upsampling 수행
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route": # 
            layers = [int(x) for x in module_def["layers"].split(",")]
            # output_filters[1:] : output_filters의 첫 요소는 input데이터의 채널인 3으로, 이는 레이어에서 나온 채널이 아니기 때문에 생략
            # output_filters[1:][i] : yolov3.cfg의 layers속성에 담겨있는 숫자들, 예를들어 -1과 61이라면 직전 레이어와 전체에서 61번째 레이어의 채널을 합쳐서 filters로 선언 (filters를 기준으로 다음 레이어의 in_channels를 정하기 때문에 반드시 필요한 과정임)
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential()) # 연산이 concatenate이므로 place holder로 nn.Sequential()사용  

        elif module_def["type"] == "shortcut":
            # yolov3.cfg의 from에는 어떤 레이어에서 skip을 수행할 것인지 명시되어있음. 예를들어 -3이면 뒤에서 세 번째 레이어에서 skip하여 현재 레이어와 합침
            # element-wise addition이므로 channel수는 그대로라서 딱히 조작을 가하지 않고 바로 filters로 선언
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential()) # 마찬가지로 연산이 element-wise addition이므로 nn.Sequential()사용

        elif module_def["type"] == "yolo": 
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # [0, 1, 2] or [3, 4, 5] or [6, 7, 8]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            # [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
            anchors = [anchors[i] for i in anchor_idxs]
            # [(10,13),  (16,30),  (33,23)] or [(30,61),  (62,45),  (59,119)] or [(116,90),  (156,198),  (373,326)]
            num_classes = int(module_def["classes"]) # 80
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules) 
        # 레이어를 생성할 때마다 이름과 구조를 nn.ModuleList()의 인스턴스인 module_list에 담음. (module_list는 dictionary타입임) 
        output_filters.append(filters) # filters : 해당 레이어의 out_channels를 뜻함. 
        # 레이어를 생성할 때마다 output_filters에 추가해서 in_channels를 설정할 때 직전 레이어의 out_channels를 사용할 수 있도록 함

        """
        cfg파일에서 yolo_layer를 나타내는 부분을 살펴보면 
        [yolo]
        mask = 6,7,8
        anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326

        mask의 의미 : anchors리스트를 구성하는 9개의 tuple중 6, 7, 8번째를 픽해서 yolo_layer로 쓰겠다는 의미. 
                    6, 7, 8번째는 (116,90),  (156,198),  (373,326)가 된다.
        
        굳이 이렇게 나눠놓은 이유는? 
        > Fully convolutional network가 13*13, 26*26, 52*52의 각기 다른 size를 갖는 Feature map을 생성하기 위해 세 갈래로 나눠지는데 
        각 size별 Feature map에 각각 b-box를 그리려면 yolo_layer도 세 갈래로 나뉘어 배치되어야 하기 때문
        13*13에 대한 yolo_layer 한 층, 26*26에 대한 yolo_layer 한 층, 52*52에 대한 yolo_layer 한 층이 각각 따로 붙는 것이다.
        
        그럼 mask 빼버리고 anchors에 필요한 튜플 세 개만 담으면 되는거 아닌가? 왜 굳이 이렇게 번거롭게 해놓았을까?
        > 아마도 anchors의 사이즈를 이리저리 조절할 때 조금 더 편하게 하기 위해서 mask를 넣은 게 아닐까? 
        
        """

    return hyperparams, module_list


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode) # size 속성을 직접 정하지 않고 scale_factor 속성에 n을 전달하면 output_size=input_size*n이 됨
        return x

class Mish(nn.Module):

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes # 80
        self.mse_loss = nn.MSELoss() 
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # 85 (x, y, w, h, confidence, class0 score, class1 score, ... , class84 score )
        self.grid = torch.zeros(1) # 밑에서 피쳐맵 사이즈에 맞춰 바꿔줌

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors) # register_buffer : 중간에 업데이트를 하지않는 일반 layer를 넣고 싶을 때 사용 (https://powerofsummary.tistory.com/158)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None
        '''
        해당 YOLOLayer가 13,26,52 중 52크기의 피쳐맵을 형성한다 가정하면
        anchor_grid
        [[[[[10, 13]]],


          [[[16, 30]]],


          [[[33, 23]]]]]
          
        anchor_grid.shape
        (1, -1, 1, 1, 2)
        
        '''


    def forward(self, x, img_size):
        stride = img_size // x.size(2) # input이미지 크기를 현재 피쳐맵의 크기로 나눈 몫을 stride로 선언
        self.stride = stride
        bs, _, ny, nx = x.shape 
        
        '''
        x.shape : (bs,255,f_size,f_size)
        bs : 배치사이즈
        255 : (=3*85) 3개의 앵커박스 * (center_x, center_y, w, h, objective score, 80개의 class score)   # YOLOLayer직전에 붙는 conv layer의 out_channels가 255
        f_size : 13 or 26 or 52  
        쉬운 해석을 위해 이하의 주석은 모두 f_size=52를 예시로 들어서 작성
        '''
        
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # contiguous : https://f-future.tistory.com/entry/Pytorch-Contiguous
        # (bs, 255, 52, 52) ==> (bs, 3, 85, 52, 52) ==> (bs, 3, 52, 52, 85)
        if not self.training: # 추론일 경우
            if self.grid.shape[2:4] != x.shape[2:4]: # 현재 grid size가 52가 아니라면
                self.grid = self._make_grid(nx, ny).to(x.device) # grid size를 52로 만듦
                # grid.shape : (1, 1, 52, 52, 2)
                # grid의 마지막 차원 크기가 2인 이유? : 52 * 52 grid의 모든 칸에 (x, y)좌표를 표현해야 하므로

            # x[..., 0:2], x[..., 2:4], x[..., 4:]를 학습시킴. [  (x,y), (w,h), (class score *80)  ]
            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride # [b_x = σ(t_x) + c_x]
            # x[..., 0:2].sigmoid()의 요소값은 0~1범위의 sigmoid값이고 self.grid의 요소값은 0~51범위의 값이므로 stride를 곱하지 않으면 백년만년 x, y 좌표값이 0~51에 머무름. stride를 곱해줘야 이미지 전체를 아우를 수 있게 됨
            
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # [b_w = p_w * e^t_w]
            # self.anchor_grid? : register_buffer의 anchor_grid가 이거인듯
            
            x[..., 4:] = x[..., 4:].sigmoid() # Pr(object) ∗ IOU(b, object) = σ(to)
            x = x.view(bs, -1, self.no) # (bs, 8112, 85)
            # 8112 : (52^2)*3

        # training=True일 때의 x.shape : (bs, 3, 52, 52, 85)
        # training=False일 때의 x.shape : (bs, 8112, 85)
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module):

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        # x.shape의 axis=2의 크기를 반환함. (-1, 3, 416, 416)라면 416을 반환하는 것
        # 이 부분을 넣은 이유는 multi-scale training으로 input 이미지의 크기를 다양하게 줬기 때문
        # multi-scale training이 가능한 이유 : 네트워크의 구조가 FC(fully connected layer)를 사용하지 않는 FCN(fully convolutional netork)인 덕분
        layer_outputs, yolo_outputs = [], [] # layer_outputs 각 layer에서 도출한 피쳐맵을 저장
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route": 
                # layers 속성의 값이 두 개일 경우 상위 레이어의 피쳐맵을 하위 피쳐맵에 channel을 기준으로 concatenate(dim=1)하는 passthrough layer
                # layers 속성의 값이 한 개일 경우 cat을 수행해도 output이 input과 같으므로 combined_outputs = layer_outputs[int(layer_i)]이 된다
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1) 
                # group_size : 합쳐진 채널의 크기 // yolov3.cfg의 groups속성값 (module_def.get("groups", 1)을 통해 default는 1로 설정)
                # group_id : group id를 부여 (default는 0)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1)) # dict.get(groups, 1) : 해당 dict의 key에 groups가 있으면 groups의 value를 불러오고 groups가 없으면 key:value = groups:1 의 요소를 하나 만들어서 삽입함
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] 
                '''
                yolo v4에서 사용되는 grouping이라고 함
                yolov3.cfg에서 groups와 groups_id 속성을 따로 설정하지 않았다면 groups=1 groups_id=0으로 자동 설정되고 
                [:, group_size * group_id : group_size * (group_id + 1)] = [:, combined_outputs.shape[1] * 0 : combined_outputs.shape[1] * 1] = [:, 0:combined_outputs.shape[1]] 이 되어서 grouping을 적용하지 않게됨
                '''
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i] # element-wise addition
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1) 
        # Darknet의 부모 클래스인 nn.Module은 self.training=True가 default이며, model.eval()일 경우 자동으로 self.training=False로 변경됨
        # torch.cat(yolo_outputs, 1) : 각각 13, 26, 52사이즈의 피쳐맵에서 예측된 b_box를 합쳐서 (bs, 10647, 85)의 predict shape를 형성
        # yolo_outputs : train일 경우 loss를 계산하기 위해서 각각 13, 26, 52사이즈의 피쳐맵에서 예측된 b_box를 리스트에 담은 형태를 유지한 채로 predict 결과를 내보냄

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def load_model(model_path, weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model
