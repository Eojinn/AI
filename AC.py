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
from sklearn.cluster import KMeans # AC의 핵심: K-Means 클러스터링
from sklearn.decomposition import PCA # <<속도 개선을 위한 PCA 임포트>>
import time # 탐지 속도 측정을 위해 time 라이브러리 임포트
from collections import Counter
import copy

# ------------------------------
# 트리거/CNN 모델 함수 (생략 - 기존과 동일)
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
    # (생략 - CustomGTSRB 클래스는 동일)
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
        return img, int(label)

class GTSRBBackdoor(Dataset):
    # (생략 - GTSRBBackdoor 클래스는 동일)
    def __init__(self, dataset, trigger_ratio=0.5, target_label=0, transform=None):
        self.dataset = dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        is_backdoor = False
        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
            is_backdoor = True
        if self.transform:
            img = self.transform(img)
        return img, int(label), is_backdoor

class TriggerOnlyDataset(Dataset):
    # <<수정: max_samples 인자를 추가하여 샘플 수 제한>>
    def __init__(self, dataset, target_label=0, transform=None, max_samples=None):
        self.dataset = dataset
        self.target_label = target_label
        self.transform = transform
        self.max_samples = max_samples # 추가된 인자
    def __len__(self):
        if self.max_samples:
            return min(len(self.dataset), self.max_samples) # 샘플 수 제한
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
        return img, label, True 

class SimpleCNN(nn.Module):
    # (생략 - SimpleCNN 클래스는 동일)
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), # AC 활성화 추출 대상 계층
            nn.Linear(128, 43)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train(model, loader, optimizer, criterion, device):
    # (생략 - train 함수는 동일)
    model.train()
    for data in loader:
        if len(data) == 3:
            imgs, labels, _ = data
        else:
            imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def extract_activations(model, dataloader, device):
    # (생략 - extract_activations 함수는 동일)
    model.eval()
    activations = []
    is_backdoor_list = [] 
    def hook(module, input, output):
        activations.append(output.detach().cpu())
    handle = model.classifier[1].register_forward_hook(hook) 
    with torch.no_grad():
        for data in dataloader:
            if len(data) == 3:
                imgs, _, is_backdoor = data 
                is_backdoor_list.extend(is_backdoor) 
            else:
                imgs, _ = data
                is_backdoor_list.extend([True] * imgs.size(0))
            imgs = imgs.to(device)
            model(imgs)
    handle.remove()
    activations_tensor = torch.cat(activations, dim=0)
    return activations_tensor.numpy(), np.array(is_backdoor_list)

# ------------------------------
# AC(활성화 클러스터링) 탐지 및 결과 출력 함수 (수정됨: PCA 추가)
# ------------------------------
def evaluate_ac_detection(model, dataloader, device, target_label, use_pca=True, pca_dims=30):
    """
    활성화 클러스터링을 수행하고, 탐지 정확도 (Recall)와 탐지 속도를 계산하여 출력합니다.
    - use_pca: PCA를 사용해 차원을 축소할지 여부
    - pca_dims: PCA 후 목표 차원 (기본 30)
    """
    start_time = time.time() # 탐지 시작 시간

    # 1. 활성화 벡터 추출
    acts, ground_truth_bool = extract_activations(model, dataloader, device)
    
    # 2. <<최적화: PCA를 통한 차원 축소>>
    if use_pca and acts.shape[1] > pca_dims:
        print(f"-> PCA 적용: {acts.shape[1]} 차원에서 {pca_dims} 차원으로 축소하여 클러스터링 속도 개선")
        pca = PCA(n_components=pca_dims)
        acts = pca.fit_transform(acts)
    
    # 3. K-Means 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(acts)
    labels_pred = kmeans.labels_ # 클러스터링 예측 레이블 (0 또는 1)
    
    # 4. 클러스터 레이블 매핑 및 탐지 예측 (생략 - 기존과 동일)
    counts = Counter(labels_pred)
    attack_cluster_label = min(counts, key=counts.get)
    prediction = (labels_pred == attack_cluster_label).astype(int)
    ground_truth = ground_truth_bool.astype(int) 
    
    # 5. TP, FN 계산 (Recall = 탐지 정확도)
    TP = np.sum((ground_truth == 1) & (prediction == 1)) 
    FN = np.sum((ground_truth == 1) & (prediction == 0)) 
    total_backdoor_samples = TP + FN

    detection_accuracy = TP / total_backdoor_samples if total_backdoor_samples > 0 else 0.0

    end_time = time.time() # 탐지 종료 시간
    detection_time = end_time - start_time

    # ----------------------------------------------------
    # 6. 결과 출력 (탐지 정확도, 탐지 속도 및 샘플당 속도)
    # ----------------------------------------------------
    time_per_sample = detection_time / total_backdoor_samples if total_backdoor_samples > 0 else 0.0
    
    print("\n" + "="*60)
    print("       AC (Activation Clustering) Detection Results ")
    print("="*60)
    print(f"[AC 탐지 정확도 (Recall)]: {detection_accuracy*100:.2f}%")
    print(f"[AC 탐지 속도 (Total)]: {detection_time:.4f} 초")
    # <<추가: 샘플당 처리 시간 출력>>
    print(f"[샘플당 탐지 속도]:      {time_per_sample * 1000:.2f} ms/샘플 ({total_backdoor_samples} 샘플 기준)")
    print("="*60)
    
    return detection_accuracy, detection_time

# ------------------------------
# 메인 실행: 모델 학습 및 AC 탐지 수행
# ------------------------------
# 경로 설정 (사용자 환경에 맞게 변경 필요)
img_dir = r"경로 설정"
label_csv = r"경로 설정"
target_label = 0 

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# <<<< 실시간 탐지 속도 최적화 설정 >>>>
N_AC_SAMPLES = 200  # 분석 샘플 수 축소 (기존 수천개에서 200개로 대폭 줄여 시간 절약)
USE_PCA = True      # PCA를 통한 클러스터링 속도 개선
PCA_DIMS = 30       # 차원 축소 목표 (128 -> 30)
BATCH_SIZE = 256    # 배치 사이즈 증가 (활성화 추출 속도 개선)

base_dataset = CustomGTSRB(img_dir, label_csv, transform=None)
# 학습 데이터: 10%의 백도어 샘플 포함
train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.1, target_label=target_label, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# AC 평가용 데이터: 샘플 수를 N_AC_SAMPLES로 제한하여 속도 개선
test_clean_dataset = CustomGTSRB(img_dir, label_csv, transform=transform)
test_backdoor_dataset = TriggerOnlyDataset(test_clean_dataset, target_label=target_label, transform=transform, max_samples=N_AC_SAMPLES)
test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n[백도어 공격에 취약한 모델 학습 중...]")
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)

# ------------------------------
# AC 탐지 실행 및 최종 출력
# ------------------------------
print(f"\n[AC 탐지 수행 중 - {N_AC_SAMPLES}개 샘플 분석]")

ac_acc, ac_time = evaluate_ac_detection(model, test_backdoor_loader, device, target_label, use_pca=USE_PCA, pca_dims=PCA_DIMS)
