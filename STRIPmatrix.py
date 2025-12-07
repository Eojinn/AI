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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # 오분류표 계산 및 시각화를 위해 추가
import matplotlib.pyplot as plt # 시각화를 위해 추가

# ------------------------------
# 상수 정의 및 실시간 탐지 최적화 설정
# ------------------------------
# 경로 (실제 데이터셋이 없는 환경에서도 실행 가능하도록 더미 경로 설정)
IMG_DIR = r"경로 설정"
LABEL_CSV = r"경로 설정"
TARGET_LABEL = 0 

# <<<< 실시간 탐지 속도 최적화 설정 >>>>
N_SAMPLES = 200 # 탐지에 사용할 샘플 수 (테스트 크기 유지)
N_PERTURBATIONS = 5 # 이미지당 섭동 횟수를 10에서 5로 축소하여 속도 개선
STRIP_THRESHOLD = 0.5 # 엔트로피 임계값 (백도어: 엔트로피 낮음, 클린: 엔트로피 높음)
NUM_CLASSES = 43

# ------------------------------
# 트리거/데이터셋/CNN 모델 함수 
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
    """GTSRB 데이터셋을 모방하는 클래스 (실제 경로가 없을 경우 더미 데이터를 사용)"""
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
            # 더미 데이터 생성 (파일이 없을 경우)
            print("[경고] CSV 파일을 찾을 수 없습니다. 더미 데이터를 사용합니다.")
            data = {'Filename': [f'{i:05d}.png' for i in range(100)], 
                    'ClassId': [i % NUM_CLASSES for i in range(100)]}
            self.labels = pd.DataFrame(data)
            
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        label = row['ClassId']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # 파일이 없으면 더미 이미지 생성
            img = Image.new('RGB', (32, 32), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class GTSRBBackdoor(Dataset):
    """백도어 공격을 위한 데이터셋 (트리거 삽입)"""
    def __init__(self, dataset, trigger_ratio=0.5, target_label=0, transform=None):
        self.dataset = dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            
        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
            
        if self.transform:
            img = self.transform(img)
        return img, int(label)

class TriggerOnlyDataset(Dataset):
    """백도어 공격 성공률(ASR) 측정을 위한 데이터셋"""
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
            
        # 클린 이미지에 트리거만 삽입
        trigger_fn = random.choice(trigger_pool)
        img = trigger_fn(img)
        
        if self.transform:
            img = self.transform(img).float()
            
        label = torch.tensor(self.target_label).long()
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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
# 학습 및 평가 함수 정의
# ------------------------------
def train(model, loader, optimizer, criterion, device):
    """모델 학습을 위한 단일 에포크 실행 함수"""
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
    """모델 평가 (정확도 계산) 함수"""
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
    # log(prob + epsilon)을 사용하여 log(0) 방지
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
    # 데이터셋이 큰 경우, 전체를 메모리에 로드하는 것은 비효율적일 수 있습니다.
    # 하지만 여기서는 N_SAMPLES가 작으므로 임시로 사용합니다.
    try:
        noise_images = [dataset[i][0].to(device) for i in range(len(dataset))]
    except Exception as e:
        print(f"[경고] 노이즈 이미지 로드 중 오류 발생: {e}. 더미 이미지를 사용합니다.")
        dummy_img = torch.rand(3, 32, 32).to(device)
        noise_images = [dummy_img] * max_samples


    start_time = time.time() # 탐지 시작 시간 측정
    
    with torch.no_grad():
        for i in range(max_samples):
            try:
                img, _ = dataset[i]
            except IndexError:
                # 더미 데이터셋 크기가 N_SAMPLES보다 작을 경우 처리
                img, _ = dataset[random.randint(0, len(dataset)-1)] 
                
            img = img.unsqueeze(0).to(device) # 원본 이미지 (1x3x32x32)
            
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

def calculate_detection_metrics(clean_entropies, backdoor_entropies, threshold=STRIP_THRESHOLD):
    """
    STRIP 탐지 성능 지표 (2x2 오분류표)를 계산합니다.
    (탐지 기준: 엔트로피 < 임계값)
    """
    # 백도어 샘플 (Actual Backdoor): 엔트로피가 낮아야 탐지됨
    backdoor_detected = sum([e < threshold for e in backdoor_entropies]) # TP (탐지 성공)
    backdoor_not_detected = sum([e >= threshold for e in backdoor_entropies]) # FN (미탐지)
    
    # 클린 샘플 (Actual Clean): 엔트로피가 높아야 거부됨
    clean_detected = sum([e < threshold for e in clean_entropies]) # FP (오탐지)
    clean_not_detected = sum([e >= threshold for e in clean_entropies]) # TN (올바른 거부)
    
    TP = backdoor_detected
    FN = backdoor_not_detected
    FP = clean_detected
    TN = clean_not_detected
    
    total_samples = TP + FN + FP + TN

    TPR = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0.0 # 백도어 탐지율 (Detection Accuracy)
    FPR = FP / (FP + TN) * 100 if (FP + TN) > 0 else 0.0 # 오탐지율 (False Positive Rate)
    
    return TP, FN, FP, TN, TPR, FPR, total_samples

# ------------------------------
# 메인 실행 및 STRIP 탐지 수행
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 데이터셋 로딩
try:
    base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
    # 학습 데이터셋 (10% 오염)
    train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.1, target_label=TARGET_LABEL, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    # 탐지 데이터셋
    test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
    test_backdoor_dataset = TriggerOnlyDataset(test_clean_dataset, target_label=TARGET_LABEL, transform=transform)
except Exception as e:
    print(f"\n[오류 발생] 데이터셋 로드 중 문제가 발생했습니다: {e}")
    print("GTSRB 데이터셋 경로(IMG_DIR, LABEL_CSV)를 확인하거나, 더미 데이터 사용에 문제가 없는지 확인하십시오.")
    exit()

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
total_test_samples = N_SAMPLES * 2

# 샘플당 처리 시간 (초)
time_per_sample = total_detection_time / total_test_samples if total_test_samples > 0 else 0.0

# 3. 탐지 성능 지표 (2x2 오분류표) 계산
TP, FN, FP, TN, TPR, FPR, total_used_samples = calculate_detection_metrics(
    clean_entropies, backdoor_entropies, threshold=STRIP_THRESHOLD
)

# --------------------------------------------------------------------------
# 🌟 혼동 행렬 시각화 코드 (2x2 매트릭스: Backdoor vs Clean) 🌟
# --------------------------------------------------------------------------

# 2x2 Confusion Matrix 구성 (Actual vs Predicted)
# Predicted: Backdoor (Detected), Predicted: Clean (Rejected)
cm_2x2 = np.array([
    [TP, FN], # Actual Backdoor
    [FP, TN]  # Actual Clean
])

# 시각화 (정수 count로 표시)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_2x2, 
    display_labels=['Predicted: Backdoor (Detected)', 'Predicted: Clean (Rejected)']
)
disp.plot(cmap='Blues', values_format='d') 

plt.title(f"STRIP Detection 2x2 Confusion Matrix (Threshold: {STRIP_THRESHOLD})")
plt.xlabel("Predicted Label (STRIP Output)")
plt.ylabel("Actual Sample Type")

# y-축 레이블을 수동으로 'Actual Backdoor'와 'Actual Clean'으로 설정
ax = plt.gca()
ax.set_yticklabels(['Actual Backdoor', 'Actual Clean'])
plt.show()

# --------------------------------------------------------------------------


# 4. 결과 출력 (오분류표 형식으로 수정)
print("\n" + "="*70)
print("                       STRIP 실시간 탐지 분석 ")
print("                      (엔트로피 임계값: %.2f)" % STRIP_THRESHOLD)
print("-" * 70)
print("Actual |    Predicted: Backdoor (Detected) |  Predicted: Clean (Rejected) |")
print("-" * 70)
print(f"Backdoor|           {TP:6d} (TP)          |           {FN:6d} (FN)         |")
print("-" * 70)
print(f"Clean |           {FP:6d} (FP)          |           {TN:6d} (TN)         |")
print("-" * 70)
print(f"\n[백도어 탐지 정확도 (TPR)]:    {TPR:.2f}% (TP / Actual Backdoor)")
print(f"[클린 오탐지율 (FPR)]:        {FPR:.2f}% (FP / Actual Clean)")
print(f"[총 탐지 소요 시간]:          {total_detection_time:.4f} 초 ({total_used_samples} 샘플 기준)")
print(f"[샘플당 평균 처리 속도]:      {time_per_sample * 1000:.2f} ms/샘플")
print("="*70)

