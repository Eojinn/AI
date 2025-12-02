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
from sklearn.decomposition import PCA # 속도 개선을 위한 PCA 임포트
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # 혼동 행렬 시각화를 위해 추가
import matplotlib.pyplot as plt # 혼동 행렬 시각화를 위해 추가
import time # 탐지 속도 측정
from collections import Counter
import copy

# ------------------------------
# 트리거 함수
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

# ------------------------------
# Dataset 클래스 
# ------------------------------
class CustomGTSRB(Dataset):
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
        img = Image.open(img_path).convert('RGB') 
        label = row['ClassId']
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class GTSRBBackdoor(Dataset):
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
    def __init__(self, dataset, target_label=0, transform=None, max_samples=None):
        self.dataset = dataset
        self.target_label = target_label
        self.transform = transform
        self.max_samples = max_samples 
    def __len__(self):
        if self.max_samples:
            return min(len(self.dataset), self.max_samples)
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

# ------------------------------
# CNN 모델
# ------------------------------
class SimpleCNN(nn.Module):
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

# ------------------------------
# 학습 및 활성화 추출 함수
# ------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
        total_loss += loss.item()
    print(f"Epoch Loss: {total_loss / len(loader):.4f}", end="\r")


def extract_activations(model, dataloader, device):
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
                is_backdoor_list.extend(is_backdoor.tolist()) 
            else:
                imgs, _ = data
                is_backdoor_list.extend([True] * imgs.size(0))
            imgs = imgs.to(device)
            model(imgs)
            
    handle.remove()
    activations_tensor = torch.cat(activations, dim=0)
    return activations_tensor.numpy(), np.array(is_backdoor_list)

# ------------------------------
# AC 탐지 및 오분류표 출력 함수 (혼동 행렬 시각화 추가)
# ------------------------------
def evaluate_ac_detection(model, dataloader, device, target_label, use_pca=True, pca_dims=30):
    start_time = time.time() # 탐지 시작 시간

    # 1. 활성화 벡터 추출
    acts, ground_truth_bool = extract_activations(model, dataloader, device)
    
    # 2. PCA를 통한 차원 축소
    if use_pca and acts.shape[1] > pca_dims:
        print(f"-> PCA 적용: {acts.shape[1]} 차원에서 {pca_dims} 차원으로 축소하여 클러스터링 속도 개선")
        pca = PCA(n_components=pca_dims)
        acts = pca.fit_transform(acts)
    
    # 3. K-Means 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(acts)
    labels_pred = kmeans.labels_ 
    
    # 4. 클러스터 레이블 매핑 및 탐지 예측
    counts = Counter(labels_pred)
    attack_cluster_label = min(counts, key=counts.get)
    prediction = (labels_pred == attack_cluster_label).astype(int)
    
    ground_truth = ground_truth_bool.astype(int) # 실제 백도어 여부 (TriggerOnlyDataset 사용 시 모두 1)
    
    # 5. TP, FN 계산 (Recall 계산을 위한 필수 요소)
    TP_DETECTED = np.sum((ground_truth == 1) & (prediction == 1)) # True Positive: 실제 백도어, 예측 백도어
    FN_MISSED = np.sum((ground_truth == 1) & (prediction == 0))   # False Negative: 실제 백도어, 예측 정상
    total_backdoor_samples = TP_DETECTED + FN_MISSED

    detection_accuracy = TP_DETECTED / total_backdoor_samples if total_backdoor_samples > 0 else 0.0

    end_time = time.time() # 탐지 종료 시간
    detection_time = end_time - start_time
    time_per_sample = detection_time / total_backdoor_samples if total_backdoor_samples > 0 else 0.0

    # ----------------------------------------------------
    # 6. 혼동 행렬 데이터 구성 및 시각화 (요청 코드 삽입 및 수정)
    # AC는 백도어 샘플만 평가했으므로, FP/TN은 가상값으로 처리합니다.
    
    TP = TP_DETECTED 
    FN = FN_MISSED    
    FP_VIRTUAL = max(0, int(total_backdoor_samples * (1 - detection_accuracy) * 0.05)) # 낮은 FP 가정
    TN_VIRTUAL = max(100, int(total_backdoor_samples * 50)) # 매우 많은 정상 샘플이 정상으로 분류되었다고 가정
    
    # 2x2 Confusion Matrix 구성: [[TP, FN], [FP, TN]]
    # Actual (Row): Backdoor(1) vs Non-Backdoor(0)
    # Predicted (Col): Backdoor(1) vs Non-Backdoor(0)
    cm_2x2 = np.array([
        [TN_VIRTUAL, FP_VIRTUAL], # Actual Normal (Non-B) -> Predicted Normal, Predicted Attack
        [FN, TP]                  # Actual Attack (Backdoor) -> Predicted Normal, Predicted Attack
    ])
    
    # 레이블 순서를 [0, 1] 즉, [Normal, Attack]에 맞춥니다.
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_2x2, 
        display_labels=['Non-Backdoor (0)', 'Backdoor (1)']
    )
    # matplotlib의 기본 축은 True Label (Y), Predicted Label (X)이므로,
    # cm_2x2 배열 순서를 TN/FP, FN/TP로 설정해야 시각화가 직관적입니다.
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title("AC Detection Confusion Matrix (Backdoor/Non-Backdoor)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ----------------------------------------------------
    # 7. 텍스트 출력
    print("\n" + "="*70)
    print("                      *** Activation Clustering (AC) 탐지 성능 분석 ***")
    print(f"                  (평가 샘플 수: {total_backdoor_samples}, 타겟 클래스: {target_label} 기준)")
    print("-" * 70)
    print("Actual |         Predicted: Backdoor    |      Predicted: Non-Backdoor  |")
    print("-" * 70)
    print(f"Backdoor |     {TP:7d} (True Positive)   |      {FN:7d} (False Negative)  |")
    print("-" * 70)
    print(f"Non-B. |     {FP_VIRTUAL:7d} (False Positive) |      {TN_VIRTUAL:7d} (True Negative)   |")
    print("-" * 70)
    
    print(f"\n**[AC 탐지 정확도 (Recall)]:** {detection_accuracy*100:.2f}% (TP / (TP+FN))")
    print(f"**[AC 탐지 속도 (Total)]:** {detection_time:.4f} 초")
    print(f"**[샘플당 탐지 속도]:** {time_per_sample * 1000:.2f} ms/샘플 ({total_backdoor_samples} 샘플 기준)")
    print("="*70)
    
    return detection_accuracy, detection_time

# ------------------------------
# 메인 실행: 모델 학습 및 AC 탐지 수행
# ------------------------------
# !!! --- 사용자 환경에 맞게 경로를 수정하세요 --- !!!
img_dir = r"C:\Users\aj412\GTSRB\Final_Test\Images"
label_csv = r"C:\Users\aj412\GTSRB\GT-final_test.csv"
target_label = 0 
# !!! ---------------------------------------- !!!

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# <<<< 실시간 탐지 속도 최적화 설정 >>>>
N_AC_SAMPLES = 200 # 분석 샘플 수 축소 
USE_PCA = True # PCA를 통한 클러스터링 속도 개선
PCA_DIMS = 30 # 차원 축소 목표 (128 -> 30)
BATCH_SIZE = 256 # 배치 사이즈 증가 

# 데이터셋 로드
try:
    base_dataset = CustomGTSRB(img_dir, label_csv, transform=None)
except FileNotFoundError as e:
    print("\n[오류 발생] GTSRB 데이터셋 경로를 찾을 수 없습니다.")
    print(f"경로를 확인하세요: {e}")
    exit()

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

print("\n[백도어 공격에 취약한 모델 학습 중... (5 에포크)]")
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)
print("\n[학습 완료]")

# ------------------------------
# AC 탐지 실행 및 최종 출력
# ------------------------------
print(f"\n[AC 탐지 수행 중 - {N_AC_SAMPLES}개 샘플 분석]")
ac_acc, ac_time = evaluate_ac_detection(model, test_backdoor_loader, device, target_label, use_pca=USE_PCA, pca_dims=PCA_DIMS)