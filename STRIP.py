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
import time # 탐지 속도 측정을 위해 time 라이브러리 추가

# ------------------------------
# 상수 정의 및 실시간 탐지 최적화 설정
# ------------------------------
# 경로 (사용자 환경에 맞게 변경 필요)
IMG_DIR = r"경로 설정"
LABEL_CSV = r"경로 설정"
TARGET_LABEL = 0 

# <<<< 실시간 탐지 속도 최적화 설정 >>>>
N_SAMPLES = 200             # 탐지에 사용할 샘플 수 (테스트 크기 유지)
N_PERTURBATIONS = 5         # 이미지당 섭동 횟수를 10에서 5로 축소
STRIP_THRESHOLD = 0.5       # 엔트로피 임계값

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
    def __init__(self, dataset, trigger_ratio=0.5, target_label=0, transform=None):
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
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------
# 학습 및 평가 함수 정의 (기존과 동일)
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
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# ------------------------------
# STRIP 방어 기법 핵심 함수 (실시간 최적화 적용)
# ------------------------------
def compute_entropy(probs):
    """주어진 확률 분포에 대한 엔트로피를 계산합니다."""
    # F.log_softmax를 사용하여 안정적인 엔트로피 계산
    log_probs = F.log_softmax(torch.log(probs + 1e-10), dim=1) 
    return -torch.sum(probs * log_probs, dim=1)

def strip_detection(model, dataset, device, n_perturbations=N_PERTURBATIONS, n_samples=N_SAMPLES):
    """
    STRIP 탐지 함수 (배치 처리 최적화).
    """
    model.eval()
    entropies = []
    
    max_samples = min(n_samples, len(dataset))
    
    # 노이즈 이미지 사전 로드 (랜덤 접근을 위해 리스트로 변환)
    noise_images = [dataset[i][0].to(device) for i in range(len(dataset))]

    start_time = time.time() # 탐지 시작 시간 측정
    
    with torch.no_grad():
        for i in range(max_samples):
            img, _ = dataset[i]
            img = img.unsqueeze(0).to(device) # 원본 이미지 (1x3x32x32)
            
            # 섭동된 이미지들을 담을 리스트 (N_PERTURBATIONS x 3 x 32 x 32)
            perturbed_batch = [] 

            for _ in range(n_perturbations):
                # 랜덤 노이즈 이미지 선택
                noise_img = random.choice(noise_images)
                
                # 믹싱 처리: M = (I + N) / 2
                mixed = (img.squeeze(0) + noise_img) / 2.0
                
                perturbed_batch.append(mixed.unsqueeze(0))

            # 섭동된 이미지들을 하나의 배치로 묶어 GPU에서 병렬 처리
            inputs = torch.cat(perturbed_batch, dim=0)
            
            # 순전파
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # 평균 엔트로피 계산
            entropy = compute_entropy(probs).mean().item()
            entropies.append(entropy)

    end_time = time.time() # 탐지 종료 시간 측정
    detection_time = end_time - start_time
    
    return entropies, detection_time

def calculate_detection_accuracy(clean_entropies, backdoor_entropies, threshold=STRIP_THRESHOLD):
    """
    STRIP 탐지 성능을 계산합니다.
    """
    # 임계값보다 엔트로피가 낮은 백도어 샘플의 수 (True Positive)
    backdoor_detected = sum([e < threshold for e in backdoor_entropies])
    
    backdoor_total = len(backdoor_entropies)
    
    # 백도어 샘플을 백도어로 올바르게 탐지할 확률 (True Positive Rate)
    tp_rate = backdoor_detected / backdoor_total * 100
    
    # 참고: 실전에서는 클린 샘플의 FPR도 함께 고려해야 합니다.
    # clean_detected = sum([e < threshold for e in clean_entropies])
    # clean_total = len(clean_entropies)
    # fp_rate = clean_detected / clean_total * 100
    
    return tp_rate

# ------------------------------
# 메인 실행 및 STRIP 탐지 수행
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 데이터셋 로딩
base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
# 학습 데이터셋 (trigger_ratio=0.1)
train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.1, target_label=TARGET_LABEL, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 학습 배치 사이즈는 64 유지
# 탐지 데이터셋
test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
test_backdoor_dataset = TriggerOnlyDataset(test_clean_dataset, target_label=TARGET_LABEL, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 모델 학습 (STRIP 탐지 전 단계)
print("\n[백도어 모델 학습 중...]")
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)

# ------------------------------
# STRIP 탐지 수행 및 측정
# ------------------------------
print(f"\n[STRIP 탐지 수행 중 - {N_SAMPLES*2}개 샘플, {N_PERTURBATIONS} 섭동]")

# 1. 클린 샘플 엔트로피 계산 및 시간 측정
clean_entropies, clean_time = strip_detection(
    model, test_clean_dataset, device, 
    n_perturbations=N_PERTURBATIONS, n_samples=N_SAMPLES
)

# 2. 백도어 샘플 엔트로피 계산 및 시간 측정
backdoor_entropies, backdoor_time = strip_detection(
    model, test_backdoor_dataset, device, 
    n_perturbations=N_PERTURBATIONS, n_samples=N_SAMPLES
)

# 총 탐지 속도
total_detection_time = clean_time + backdoor_time
total_samples = N_SAMPLES * 2

# 샘플당 처리 시간 (초)
time_per_sample = total_detection_time / total_samples if total_samples > 0 else 0.0

# 3. 탐지 정확도 (TPR) 계산
detection_accuracy = calculate_detection_accuracy(
    clean_entropies, backdoor_entropies, threshold=STRIP_THRESHOLD
)

# ------------------------------
# 결과 출력
# ------------------------------
print("\n" + "="*60)
print("             *** STRIP 실시간 탐지 결과 요약 ***")
print("="*60)
print(f"[STRIP 탐지 정확도 (TPR)]:  {detection_accuracy:.2f}% (임계값: {STRIP_THRESHOLD})")
print(f"[총 탐지 소요 시간]:         {total_detection_time:.4f} 초 ({total_samples} 샘플 기준)")
print(f"[샘플당 평균 처리 속도]:     {time_per_sample * 1000:.2f} ms/샘플")

print("="*60)
