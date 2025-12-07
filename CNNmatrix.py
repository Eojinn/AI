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
import torch.nn.functional as F
import time 
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion Matrix 계산을 위해 추가
import matplotlib.pyplot as plt

# ------------------------------
# 상수 정의 및 최적화 설정
# ------------------------------
# 경로 (사용자 환경에 맞게 변경 필요)
# !!! 중요: 이 경로를 실제 GTSRB 데이터셋 경로로 변경해야 코드가 실행됩니다. !!!
IMG_DIR = r"경로 설정" 
LABEL_CSV = r"경로 설정"
TARGET_LABEL = 0 

# <<<< 실시간 추론 속도 최적화 설정 >>>>
BATCH_SIZE = 256 
NUM_CLASSES = 43 # GTSRB 클래스 수

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
    def __init__(self, img_dir, label_csv, transform=None):
        self.img_dir = img_dir
        try:
            # --- [실제 CSV 로딩 로직] ---
            df = pd.read_csv(label_csv, sep=';')
            df = df.dropna(subset=['ClassId'])
            df['ClassId'] = df['ClassId'].astype(int)
            self.labels = df.reset_index(drop=True)
            # ---------------------------
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {label_csv}. 경로를 확인하세요.")
            
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        
        if not os.path.exists(img_path):
            # 파일이 없는 경우, 무작위로 다른 샘플을 반환
            return self.__getitem__(random.randint(0, len(self.labels) - 1))

        label = row['ClassId']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class GTSRBBackdoor(Dataset):
    def __init__(self, dataset, trigger_ratio=0.1, target_label=0, transform=None):
        self.dataset = dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple) and len(data) == 2:
            img, label = data
        else:
            raise ValueError(f"Base Dataset의 __getitem__이 예상치 않은 형식의 데이터를 반환했습니다: {data}")

        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
        if self.transform:
            img = self.transform(img)
        return img, int(label)

# TriggerOnlyDataset은 순수 추론 분석에서는 사용하지 않습니다.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES) # 43개 클래스
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

# ------------------------------
# 탐지 기능 (순수 추론 시간 측정 및 오분류표 형식 출력으로 수정)
# ------------------------------
def run_detection_analysis(model, train_loader, test_clean_loader, device):
    """
    CNN 모델을 학습시킨 후, 클린 데이터셋에 대한 순수 추론 시간과 분류 성능을 측정하고,
    결과를 특정 클래스(0)를 중심으로 묶은 2x2 오분류표 형식으로 출력하고 시각화합니다.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 1. 모델 학습 (탐지 전 처리 과정)
    print("\n[모델 학습 중 (탐지 전처리 과정)...]")
    start_train_time = time.time()
    for epoch in range(5):
        train(model, train_loader, optimizer, criterion, device)
    end_train_time = time.time()
    print(f"[학습 완료] 소요 시간: {end_train_time - start_train_time:.2f} 초")
        
    # 2. 순수 추론 성능 측정 및 오분류표 데이터 수집
    model.eval()
    all_labels = []
    all_preds = []
    
    # 추론 시간 측정 시작
    start_time = time.time()
    
    with torch.no_grad():
        for imgs, labels in test_clean_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # 추론 시간 측정 종료
    end_time = time.time()
    detection_time = end_time - start_time
    
    total_samples = len(all_labels)
    
    # 3. 결과 계산 (클린 정확도 및 샘플당 시간)
    correct = np.sum(np.array(all_labels) == np.array(all_preds))
    detection_accuracy = 100.0 * correct / total_samples
    time_per_sample = detection_time / total_samples if total_samples > 0 else 0.0
    
    # 4. 오분류표 (Confusion Matrix) 계산 (43x43)
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    
    # 타겟 클래스 (Class 0: 속도 제한 20) 기준의 2x2 오분류표 지표 추출
    target_class = TARGET_LABEL
    
    # TP: 실제 0, 예측 0 (올바른 분류)
    TP_TARGET = cm[target_class, target_class] 
    
    # FN: 실제 0, 예측 Other (타겟 미분류)
    FN_TARGET = np.sum(cm[target_class, :]) - TP_TARGET 
    
    # FP: 실제 Other, 예측 0 (오분류)
    FP_TARGET = np.sum(cm[:, target_class]) - TP_TARGET 
    
    # TN: 실제 Other, 예측 Other (올바른 분류)
    TN_TARGET = total_samples - TP_TARGET - FN_TARGET - FP_TARGET 

    # --------------------------------------------------------------------------
    # 🌟 혼동 행렬 시각화 코드 (2x2 매트릭스: 타겟 0 vs 나머지) 🌟
    # --------------------------------------------------------------------------
    
    # 2x2 Confusion Matrix 구성
    # True Label (Row): Class 0 vs Other
    # Predicted Label (Col): Class 0 vs Other
    cm_2x2 = np.array([
        [TP_TARGET, FN_TARGET], # 실제 0 (Target)
        [FP_TARGET, TN_TARGET]  # 실제 Other (Non-Target)
    ])

    # 시각화 (정수 count로 표시)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_2x2, 
        display_labels=[f'Class {TARGET_LABEL} (Target)', 'Other Classes']
    )
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title(f"2x2 Confusion Matrix (Target Class {TARGET_LABEL} vs Others)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # --------------------------------------------------------------------------
    
    # 5. 결과 출력 (오분류표 형식)
    print("\n" + "="*70)
    print("                       CNN 순수 추론 분류 성능 분석 ")
    print(f"                      (클린 데이터셋, 타겟 클래스 {TARGET_LABEL} 기준)")
    print("-" * 70)
    print(f"       |      Predicted: Class {TARGET_LABEL} |       Predicted: Other      |")
    print("-" * 70)
    # TP_TARGET과 FN_TARGET은 실제 Class 0에 대한 예측 결과입니다. (ConfusionMatrixDisplay와 일치시키기 위해 레이블 조정)
    print(f"Actual |      {TP_TARGET:7d} (True Positive)    |       {FN_TARGET:7d} (False Negative)  |")
    print(f"Class {TARGET_LABEL}|")
    print("-" * 70)
    # FP_TARGET과 TN_TARGET은 실제 Other Class에 대한 예측 결과입니다.
    print(f"Actual |      {FP_TARGET:7d} (False Positive)   |       {TN_TARGET:7d} (True Negative)   |")
    print(f"Other |")
    print("-" * 70)
    
    # 6. 추론 속도 및 정확도 요약 출력
    print(f"\n[전체 클린 정확도]: {detection_accuracy:.2f}%")
    print(f"[총 추론 시간]: {detection_time:.4f} 초")
    print(f"[샘플당 추론 속도]: {time_per_sample * 1000:.2f} ms/샘플 ({total_samples} 샘플 기준)")
    print("="*70)
    
    return detection_accuracy, detection_time

# ------------------------------
# 메인 실행
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 데이터셋 로드
try:
    base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
    test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
except FileNotFoundError as e:
    print("\n[오류 발생] GTSRB 데이터셋 경로를 찾을 수 없습니다.")
    print(f"경로를 확인하세요: {e}")
    exit()

# 학습 데이터: 10%의 백도어 샘플 포함
train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.1, target_label=TARGET_LABEL, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 클린 테스트 데이터: 순수 추론 성능 측정용
test_clean_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# 탐지 분석 실행 (CNN의 순수 분류 성능 측정)
run_detection_analysis(model, train_loader, test_clean_loader, device)

