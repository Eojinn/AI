# -*- coding: utf-8 -*- 
# ------------------------------
# 필요한 라이브러리 임포트
# ------------------------------
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageDraw
import random
from torchvision import transforms
import time # 탐지 속도 측정을 위해 time 라이브러리 추가

# ------------------------------
# 상수 정의 및 최적화 설정
# ------------------------------
# 경로 (사용자 환경에 맞게 변경 필요)
IMG_DIR = r"경로 설정"
LABEL_CSV = r"경로 설정"
TARGET_LABEL = 0 

# <<<< 실시간 추론 속도 최적화 설정 >>>>
BATCH_SIZE = 256 # 배치 사이즈를 64에서 256으로 늘려 GPU 활용 및 추론 속도 개선

# ------------------------------
# 트리거/데이터셋/CNN 모델 함수 (기존 코드와 동일)
# ------------------------------
def red_square(img):
    img = img.convert('RGB')
    np_img = np.array(img)
    h, w = np_img.shape[:2]
    np_img[h-5:h, w-5:w] = [255, 0, 0]
    return Image.fromarray(np_img)

def blue_circle(img):
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.ellipse((5, 5, 15, 15), fill=(0, 0, 255))
    return img

def yellow_cross(img):
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.line((0, 0, 15, 15), fill=(255, 255, 0), width=2)
    draw.line((15, 0, 0, 15), fill=(255, 255, 0), width=2)
    return img

def white_dots(img):
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        x, y = random.randint(0, 31), random.randint(0, 31)
        draw.point((x, y), fill=(255, 255, 255))
    return img

trigger_pool = [red_square, blue_circle, yellow_cross, white_dots]

class CustomGTSRB(Dataset):
    # (생략 - 클래스 정의는 기존과 동일)
    def __init__(self, img_dir, label_csv, transform=None):
        self.img_dir = img_dir
        df = pd.read_csv(label_csv, sep=';')
        df = df.dropna(subset=['ClassId'])
        df['ClassId'] = df['ClassId'].astype(int)
        self.labels = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        label = row['ClassId']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class GTSRBBackdoor(Dataset):
    # (생략 - 클래스 정의는 기존과 동일)
    def __init__(self, dataset, trigger_ratio=0.1, target_label=0, transform=None):
        self.dataset = dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class TriggerOnlyDataset(Dataset):
    # (생략 - 클래스 정의는 기존과 동일)
    def __init__(self, dataset, target_label=0, transform=None):
        self.dataset = dataset
        self.target_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        trigger_fn = random.choice(trigger_pool)
        img = trigger_fn(img)
        if self.transform:
            img = self.transform(img).float()
        label = torch.tensor(self.target_label).long()
        return img, label

class SimpleCNN(nn.Module):
    # (생략 - 클래스 정의는 기존과 동일)
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 43)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------
# 학습 및 평가 함수 (기존과 동일)
# ------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    # 이 함수는 run_detection_analysis 내에서 통합되어 사용됩니다.
    pass

# ------------------------------
# 탐지 기능 (모델 학습 시간을 제외한 순수 추론 시간 측정으로 수정)
# ------------------------------
def run_detection_analysis(model, train_loader, test_clean_loader, device):
    """
    CNN 모델을 학습시킨 후, 클린 데이터셋에 대한 순수 추론 시간과 정확도를 측정합니다.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 1. 모델 학습 (탐지 전 처리 과정으로 간주하며, 시간 측정 대상에서 제외)
    print("\n[모델 학습 중 (탐지 전처리 과정)...]")
    for epoch in range(5):
        train(model, train_loader, optimizer, criterion, device)
        
    # 2. 순수 추론 성능 측정
    model.eval()
    correct, total = 0, 0
    
    # 추론 시간 측정 시작
    start_time = time.time()
    
    with torch.no_grad():
        for imgs, labels in test_clean_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    # 추론 시간 측정 종료
    end_time = time.time()
    detection_time = end_time - start_time
    
    # 3. 결과 계산
    detection_accuracy = 100.0 * correct / total
    
    # 실시간 성능 지표: 샘플당 처리 시간 (ms)
    time_per_sample = detection_time / total if total > 0 else 0.0
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("       CNN 순수 추론 성능 분석 결과 ")
    print("      (백도어 탐지 기법 없이 기반 모델의 예측 속도만 측정)")
    print("="*60)
    print(f"[탐지 정확도 (클린 정확도)]:    {detection_accuracy:.2f}%")
    print(f"[총 추론 시간]:              {detection_time:.4f} 초")
    print(f"[샘플당 추론 속도]:          {time_per_sample * 1000:.2f} ms/샘플 ({total} 샘플 기준)")
    print("="*60)
    
    return detection_accuracy, detection_time

# ------------------------------
# 메인 실행 및 탐지 분석
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.1, target_label=TARGET_LABEL, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
test_clean_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False) # 배치 사이즈 최적화 적용

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# 탐지 분석 실행

run_detection_analysis(model, train_loader, test_clean_loader, device)
