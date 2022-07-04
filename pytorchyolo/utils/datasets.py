from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize(image, size): # 이미지를 원하는 사이즈로 보간하는 함수
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0) # unsqueeze(0) : https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
    return image
    # F.interpolate : axis=2 이상인 차원의 shape를 size옵션 만큼 변환함
    # 예를들어 torch.Size([1, 3, 2, 2, 2])의 5차원 텐서에 F.interpolate(size=4)를 적용하면 torch.Size([1, 3, 4, 4, 4])로 바뀜
    # 그러나 image는 3차원 텐서라서 axis=2이상인 차원이 한 개 뿐이라 우리가 원하는 이미지의 가로세로 사이즈(두 차원)의 변환이 불가능함
    # 따라서 image에 unsqueeze(0)을 적용해서 4차원으로 늘려서 F.interpolate를 적용한 후 squeeze(0)으로 다시 3차원으로 바꿈
    
    # 왜 굳이 axis=2 이상인 차원만 바꿀 수 있게 만들어 놓은 건지는 모르겠음

class ImageFolder(Dataset): 
    '''
    추론에 사용(detect.py)
    
    '''
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # 추론용 데이터라 b_box데이터가 없으므로 (class, x, y, w, h) 대신 영행렬(0, 0, 0, 0, 0)로 대체
        boxes = np.zeros((1, 5)) 

        # transform 적용
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)
    
    


class ListDataset(Dataset):
    '''
    학습 및 평가에 사용(train.py, test.py)
    이미지 사이즈가 다른 경우가 있을 수 있으므로 collate_fn 메소드를 클래스 내부에 생성하여 dataloader만들 때 사용
    
    '''
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        '''
        list_path : data/coco/trainvalno5k.txt 또는 data/coco/trainvalno5k.txt
        multiscale : augmentation의 일환으로 input이미지의 사이즈를 랜덤하게 바꿈
        transform : train일 경우 transforms.py의 AUGMENTATION_TRANSFORMS
                    test일 경우 transforms.py의 DEFAULT_TRANSFORMS
        '''
        with open(list_path, "r") as file:
            self.img_files = file.readlines() # img_files에 이미지 경로들을 담음

        self.label_files = [] # label파일을 담을 리스트 선언
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1)) # 경로의 images을 labels로 바꿔서 label 경로 생성
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path)) # os.path.basename(path) : 경로(path)에서 폴더 부분 다 버리고 파일명만 남김
            label_file = os.path.splitext(label_file)[0] + '.txt' # os.path.splitext(file) : 파일명(file)에서 '.'을 기준으로 split
            self.label_files.append(label_file) # label파일을 label_files에 삽입

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip() # index % len(self.img_files) : index가 계속 커지다가 파일 전체 개수와 같아지면 다시 index 0으로 바꿔서 index out of range 오류를 방지하려는 목적인듯

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # 파일이 비어있을 경우, warning 문구를 생략하는 대신 밑의 exception 문구를 띄움
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # batch 내에 비어있는 데이터가 있을 경우 삭제
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # multiscale=True일 때 10 batch마다 이미지 사이즈를 변경
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # 이미지의 가로세로 크기가 img_size가 되도록 보간
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

# ==========================================================================================

from pytorchyolo.utils.transforms import AUGMENTATION_TRANSFORMS
from torch.utils.data import DataLoader
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.parse_config import parse_data_config
from easydict import EasyDict

def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):

    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

def run():
    print('start!!!!!!!!!')
    args = EasyDict({'model':'config/yolov3.cfg',
                    'data':'config/coco.data',
                    'epochs':300,
                    'verbose':True,
                    'n_cpu':8,
                    'pretrained_weights':'weights/darknet53.conv.74',
                    'checkpoint_interval':1,
                    'evaluation_interval':1,
                    'multiscale_training':True,
                    'iou_thres':0.5,
                    'conf_thres':0.1,
                    'nms_thres':0.5,
                    'logdir':'logs'
                })

    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.pretrained_weights)
    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    for _, imgs, targets in dataloader:
        print(targets.shape)

if __name__ == '__main__':
    run()
