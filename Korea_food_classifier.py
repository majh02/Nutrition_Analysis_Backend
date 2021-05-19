import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
from matplotlib import font_manager, rc

CUDA_LAUNCH_BLOCKING = 1

font_path = "C:\\Users\\shin yu jin\\PycharmProjects\\food_Classifier\\NanumSquareL.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()

rc('font', family=font)

device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

data_dir = './data/korea_food'  # 압축 해제된 데이터셋의 디렉토리 경로

batch_size = 100
num_epochs = 50
learning_rate = 0.0001


class FoodDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*', '*')))
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.all_data[index]
        img = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = os.path.basename(data_path)

        if len(label) == 11:
            label = int(label[1:2])
        elif len(label) == 12:
            label = int(label[1:3])
        else:
            label = int(label[1:4])

        return img, label

    def __len__(self):
        length = len(self.all_data)
        return length


class InputDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.all_data = glob.glob(os.path.join(data_dir, mode, '*'))
        self.transform = transform
        # print(self.all_data)

    def __getitem__(self, index):
        data_path = self.all_data[0]

        img = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = 0

        return img, label

    def __len__(self):
        length = len(self.all_data)
        return length


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(120, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([120, 120]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data = FoodDataset(data_dir='./data/korea_food', mode='train', transform=data_transforms['train'])

val_data = FoodDataset(data_dir='./data/korea_food', mode='val', transform=data_transforms['val'])

test_data = FoodDataset(data_dir='./data/korea_food', mode='test', transform=data_transforms['val'])

input_data = InputDataset(data_dir='./data/korea_food', mode='input', transform=data_transforms['val'])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
input_loader = DataLoader(input_data, batch_size=batch_size, shuffle=False, drop_last=True)


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


def train(num_epochs, model, data_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_loss = 9999999
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax).float().mean()

            if (i + 1) % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                    epoch + 1, num_epochs, i + 1, len(data_loader), loss.item(), accuracy.item() * 100))

        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir)


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
        print('Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, correct / total * 100, avrg_loss))
    model.train()
    return avrg_loss


def test(model, data_loader, device):
    print('Start test..')
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, argmax = torch.max(outputs, 1)  # max()를 통해 최종 출력이 가장 높은 class 선택
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))
    model.train()


def save_model(model, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    output_path = os.path.join(saved_dir, file_name)

    torch.save(check_point, output_path)

torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정

model = SimpleCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model = model.to(device)
val_every = 1
saved_dir = './saved/SimpleCNN'

model_path = './saved/SimpleCNN/best_model.pt'
model = SimpleCNN().to(device)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['net']
model.load_state_dict(state_dict)


test(model, test_loader, device)

food = ['김치볶음밥', '떡볶이', '라면', '만두', '미역국', '배추김치', '삼겹살', '시금치나물', '잡곡밥', '잡채', '훈제오리', '갈비구이',
        '갈치구이', '고등어구이', '곱창', '닭갈비', '떡갈비', '불고기', '장어구이', '조개구이', '더덕구이', '김치찌개', '된장찌개', '순두부찌개'
    , '갈비탕', '감자탕', '설렁탕', '매운탕', '삼계탕', '추어탕', '계란국', '떡국/만두국', '무국', '육개장', '콩나물국', '양념치킨', '후라이드치킨'
    , '피자', '새우튀김', '고추튀김', '보쌈', '간장게장', '양념게장', '깻잎장아찌', '계란후라이', '김밥', '비빔밥', '누룽지', '알밥', '유부초밥']
ret = list(enumerate(food))


model.eval()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

input_img = input_data[0][0].unsqueeze(dim=0).to(device)
output = model(input_img)
_, argmax = torch.max(output, 1)  ##가장 높은 값을 갖는 인덱스(index) 하나를 뽑음
pred = argmax.int()
label = input_data[0][1]



for i, body in enumerate(food):
    if pred == i:
        idx = i
        pred_title = body
        break


def return_label():
    return str(idx)


print(f'라벨은 {return_label()}입니다.')