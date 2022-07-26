

def parse_model_config(path):
    '''
    path : [workspace]/config/coco.data  # 모델의 구조와 하이퍼파라미터를 담은 cfg파일
    
    '''
    file = open(path, 'r', encoding="UTF-8")
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] 
    module_defs = []
    for line in lines:
        if line.startswith('['):  # '[' = yolov3.cfg 파일에 [convolutional], [yolo] 이런식으로 대괄호로 감싸져있는 부분이 있는데 각 블럭을 대표하는 속성을 문자열로 표현한 것임
            module_defs.append({}) # for문 돌 때마다 딕셔너리가 하나씩 생성됨
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional': # type key의 value가 convolutional이라면 batch_normalize key를 생성하여 value를 0으로 설정
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=") # 위의 if가 대괄호가 블럭의 이름표를 만든다면 else는 실제 블럭을 구성할 때 사용하는 옵션을 딕셔너리에 추가함
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    '''
    path : [workspace]/config/coco.data  # 데이터셋에 대한 종합적인 정보
    
    '''
    options = dict()
    options['num_workers'] = '2'
    with open(path, 'r', encoding="UTF-8") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
