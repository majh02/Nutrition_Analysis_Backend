import flask
import werkzeug
from werkzeug.utils import secure_filename
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob

app = flask.Flask(__name__)
device = 'cpu'

##서버에 업로드된 이미지 데이터셋
class InputDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.all_data = glob.glob(os.path.join(data_dir, mode, '*'))
        self.transform = transform
        #print(self.all_data)

    def __getitem__(self, index):
        data_path = self.all_data[0]

        img = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = 0

        return img, label

    def __len__(self):
        length = len(self.all_data)
        print(length)
        return length

##SimpleCNN 모델 레이어계층
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 50)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

@app.route('/', methods=['POST'])
def handle_request():

    ##서버에 업로드된 이미지 저장
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)

		##모델의 input폴더에 저장
    imagefile_path = './data/korea_food/input'
    imagefile.save(os.path.join(imagefile_path, filename))

    ##모델 불러오기
    predicted_label = model_process()

    #print(predicted_label)

    return predicted_label

##입력된 이미지에 대한 모델처리 결과 라벨
def model_process():
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize([120, 120]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    input_data = InputDataset(data_dir='./', mode='input', transform=data_transforms['val'])

    model_path = './saved/SimpleCNN/best_model.pt'
    model = SimpleCNN().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict)
    model.eval()

    food = ['김치볶음밥', '떡볶이', '라면', '만두', '미역국', '배추김치', '삼겹살', '시금치나물', '잡곡밥', '잡채', '훈제오리', '갈비구이',
            '갈치구이', '고등어구이', '곱창', '닭갈비', '떡갈비', '불고기', '장어구이', '조개구이', '더덕구이', '김치찌개', '된장찌개', '순두부찌개'
        , '갈비탕', '감자탕', '설렁탕', '매운탕', '삼계탕', '추어탕', '계란국', '떡국/만두국', '무국', '육개장', '콩나물국', '양념치킨', '후라이드치킨'
        , '피자', '새우튀김', '고추튀김', '보쌈', '간장게장', '양념게장', '깻잎장아찌', '계란후라이', '김밥', '비빔밥', '누룽지', '알밥', '유부초밥']

    input_img = input_data[0][0].unsqueeze(dim=0).to(device)
    output = model(input_img)
    _, argmax = torch.max(output, 1)  ##가장 높은 값을 갖는 인덱스(index) 하나를 뽑음
    pred = argmax.int()
    label = input_data[0][1]

    for i, body in enumerate(food):
        if pred == i:
            idx = i
            pred_title = body
            print(pred_title)
            break

    print("추측 라벨값=" + str(idx))

    return str(idx)


app.run(host="192.168.56.1", port=5000, debug=True)