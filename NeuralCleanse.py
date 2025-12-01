# -*- coding: utf-8 -*- 
# ------------------------------
# 필요한 라이브러리 임포트 (NC에 필요한 라이브러리 추가/수정)
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
import copy # 모델 복사를 위해 copy 라이브러리 추가
# --- NC Confusion Matrix 시각화를 위해 추가 ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ---------------------------------------------

# ------------------------------
# 상수 정의
# ------------------------------
# !!! --- 사용자 환경에 맞게 경로를 수정하세요 --- !!!
# GTSRB 데이터셋의 실제 경로로 변경해야 합니다.
IMG_DIR = r"실제 경로"
LABEL_CSV = r"실제 경로"
TARGET_LABEL = 0 # 백도어 공격의 타겟 레이블
# !!! ---------------------------------------- !!!

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
    def __init__(self, img_dir, label_csv, transform=None):
        self.img_dir = img_dir
        try:
            df = pd.read_csv(label_csv, sep=';')
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {label_csv}. 경로를 확인하세요.")

        df = df.dropna(subset=['ClassId'])
        df['ClassId'] = df['ClassId'].astype(int)
        self.labels = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        if not os.path.exists(img_path):
            # 파일이 없는 경우, 건너뛰고 오류 처리
            print(f"경고: 이미지를 찾을 수 없습니다: {img_path}")
            # 무작위로 다른 샘플을 반환하거나 오류를 발생시켜야 하지만, 여기서는 임시로 첫 번째 샘플을 반환
            return self.__getitem__(random.randint(0, len(self.labels) - 1))

        label = row['ClassId']
        img = Image.open(img_path).convert('RGB')
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
        # Base Dataset의 getitem이 튜플을 반환하도록 처리
        data = self.dataset[idx]
        if isinstance(data, tuple) and len(data) == 2:
            img, label = data
        else:
            # 예상치 못한 형식의 데이터가 반환될 경우
            raise ValueError(f"Base Dataset의 __getitem__이 예상치 않은 형식의 데이터를 반환했습니다: {data}")

        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
        if self.transform:
            img = self.transform(img)
        return img, int(label)

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
            nn.Linear(128, 43)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch Loss: {total_loss / len(loader):.4f}", end="\r")

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

# ---------------------------------------
# Neural Cleanse (NC) 핵심 로직
# ---------------------------------------

def reverse_engineer_trigger(model, class_id, dataloader, device, max_iter=1000, lr=0.005):
    """
    Neural Cleanse의 핵심: 특정 타겟 클래스로 오분류하게 만드는 보편적인 트리거와 마스크를 역공학합니다.
    """
    # 모델을 복사하여 역공학 과정이 원본 모델에 영향을 주지 않도록 합니다.
    temp_model = copy.deepcopy(model)
    temp_model.eval()
    
    # 마스크와 트리거 (패턴) 초기화
    mask = torch.rand(3, 32, 32, device=device).requires_grad_(True)
    pattern = torch.rand(3, 32, 32, device=device) - 0.5
    pattern.requires_grad_(True)

    optimizer = torch.optim.Adam([mask, pattern], lr=lr)
    target_label = torch.tensor([class_id], device=device).long()
    
    # 원본 이미지 선택 (클린 이미지 100개만 사용)
    clean_images = []
    
    # 데이터 로더를 통해 클린 이미지 수집
    for imgs, labels in dataloader:
        # 타겟 클래스가 아닌 이미지 100개만 사용
        non_target_imgs = imgs[labels != class_id]
        
        # 이미 수집된 이미지 + 현재 배치의 non_target 이미지 수가 100개 이하인지 확인
        remaining_slots = 100 - sum(img.size(0) for img in clean_images)
        if remaining_slots > 0:
            clean_images.append(non_target_imgs[:remaining_slots].to(device))
        
        if sum(img.size(0) for img in clean_images) >= 100:
            break
            
    if not clean_images:
        # print(f"클래스 {class_id}에 대한 비타겟 클린 이미지를 찾을 수 없습니다. NC 탐색 건너뛰기.") # 출력 제거
        return 0, 0, 0, 0.0

    clean_images = torch.cat(clean_images, dim=0)[:100]

    # 손실 함수
    criterion_conf = nn.CrossEntropyLoss() 
    
    # 역공학
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # 트리거 삽입: X_p = (1 - M) * X + M * P
        # 패턴 P를 클린 이미지의 평균에 맞추어 트리거의 색상 변화를 제어합니다. (NC 논문의 구현 참고)
        # 텐서 형태에 맞게 조정 (100x3x32x32)
        pattern_applied = pattern.unsqueeze(0).repeat(clean_images.size(0), 1, 1, 1)
        masked_image = (1 - mask) * clean_images + mask * (pattern_applied + clean_images.mean(dim=(0, 2, 3), keepdim=True))

        # 예측
        output = temp_model(masked_image)
        
        # 1. 공격 성공 손실 (타겟 레이블로 예측되도록 유도)
        loss_attack = criterion_conf(output, target_label.repeat(masked_image.size(0)))
        
        # 2. 트리거 크기 손실 (마스크 M의 L1 정규화)
        # 마스크 M의 값이 0에 가까워지도록 유도 -> 트리거 크기를 작게 유지
        loss_size = torch.sum(torch.abs(mask))
        
        # 총 손실 (가중치 L_2=0.01)
        loss = loss_attack + 0.01 * loss_size
        
        loss.backward()
        optimizer.step()
        
        # 마스크 클리핑 (유효 범위 0~1 유지)
        with torch.no_grad():
            mask.clamp_(0, 1)

    # 최종 이상치 점수 계산: 트리거 크기 (마스크의 L1 노름)
    anomaly_score = torch.sum(mask).item()
    
    # 수렴 후 공격 성공률 계산
    with torch.no_grad():
        final_output = temp_model(masked_image)
        preds = final_output.argmax(1)
        asr = (preds == target_label.repeat(masked_image.size(0))).float().mean().item()

    return anomaly_score, mask, pattern, asr


def run_neural_cleanse(model, dataloader, num_classes=43, device=None, threshold_multiplier=2.0):
    """
    Neural Cleanse 탐지 프로세스를 실행하고 탐지 정확도를 계산하며, 혼동 행렬을 시각화합니다.
    """
    print("\n[Neural Cleanse 탐지 시작...]")
    start_time = time.time() # 탐지 시작 시간

    anomaly_scores = []
    
    # 1. 모든 클래스에 대해 잠재적 트리거 역공학 수행
    for class_id in range(num_classes):
        # 역공학 과정 중 로그는 생략하여 출력을 간소화합니다.
        score, _, _, asr = reverse_engineer_trigger(model, class_id, dataloader, device)
        anomaly_scores.append(score)
        
    anomaly_scores = np.array(anomaly_scores)
    
    # 2. 이상치 점수 계산 및 탐지
    # 중앙값(median)과 MAD(Median Absolute Deviation)를 사용하여 임계값 계산
    median = np.median(anomaly_scores)
    mad = np.median(np.abs(anomaly_scores - median))
    
    if mad == 0:
        threshold = np.inf
    else:
        # 임계값 설정: 중앙값 + 임계값_배수 * MAD
        threshold = median + threshold_multiplier * mad

    # 탐지 예측: 점수가 임계값을 초과하면 백도어가 심어진 것으로 예측
    is_backdoor_detected = anomaly_scores > threshold
    
    # 3. 2x2 Confusion Matrix 구성
    
    # True Positive (TP): 실제 백도어 클래스가 백도어로 탐지됨 (1 또는 0)
    TP = 1 if is_backdoor_detected[TARGET_LABEL] else 0
    
    # False Negative (FN): 실제 백도어 클래스가 정상으로 오인됨 (1 - TP)
    FN = 1 - TP 

    # False Positive (FP): 실제 정상 클래스(비타겟)가 백도어로 오인됨
    # TARGET_LABEL을 제외한 나머지 클래스에서 탐지된 수
    non_target_detections = np.delete(is_backdoor_detected, TARGET_LABEL)
    FP = np.sum(non_target_detections).item()

    # True Negative (TN): 실제 정상 클래스(비타겟)가 정상으로 탐지됨
    TN = (num_classes - 1) - FP
    
    # Confusion Matrix (True Label, Predicted Label)
    # cm_2x2 = [[TN, FP], [FN, TP]] where 0=Non-Backdoor, 1=Backdoor
    cm_2x2 = np.array([
        [TN, FP], # Actual Non-Backdoor (0) -> Predicted Non-B, Predicted B
        [FN, TP]  # Actual Backdoor (1) -> Predicted Non-B, Predicted B
    ])
    
    # Recall/TPR (탐지 정확도)
    detection_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0.0 
    
    end_time = time.time() # 탐지 종료 시간
    detection_time = end_time - start_time
    detection_time_ms = detection_time * 1000

    # ----------------------------------------------------
    # 4. 혼동 행렬 시각화
    # ----------------------------------------------------
    print("\n[Confusion Matrix 시각화]")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_2x2, 
        display_labels=['Non-Backdoor (0)', 'Backdoor (1)']
    )
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title("NC Detection Confusion Matrix (Backdoor vs Non-Backdoor)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ----------------------------------------------------
    # 5. 결과 출력 
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("                      *** Neural Cleanse (NC) Detection Results ***")
    print(f"                       (총 {num_classes}개 클래스에 대해 탐색 완료)")
    print("-" * 80)
    
    print(f"Anomaly Scores (L1 노름): {anomaly_scores.round(2)}")
    print(f"Median: {median:.2f}, MAD: {mad:.2f}, Threshold (Median + {threshold_multiplier:.1f}*MAD): {threshold:.2f}")
    print("-" * 80)

    # 행렬 값 출력
    print("                      *** 2x2 NC Detection Matrix ***")
    print("-" * 80)
    print(f"       | Predicted Normal (0) | Predicted Attack (1) |")
    print("-" * 80)
    print(f"Actual | {TN:20d} (True Negative) | {FP:17d} (False Positive) |")
    print(f"Normal |")
    print("-" * 80)
    print(f"Actual | {FN:20d} (False Negative) | {TP:17d} (True Positive) |")
    print(f"Attack |")
    print("-" * 80)

    # 타겟 클래스 (TARGET_LABEL)의 탐지 결과 강조
    target_score = anomaly_scores[TARGET_LABEL]
    target_detection = "탐지 성공 (Detected!)" if TP == 1 else "탐지 실패 (Missed)"
    
    print(f"-> 타겟 클래스 {TARGET_LABEL} (실제 백도어): 점수 {target_score:.2f} ({target_detection})")
    
    # 이상치로 분류된 모든 클래스 출력
    detected_classes = np.where(is_backdoor_detected)[0]
    print(f"-> 이상치로 탐지된 클래스: {detected_classes.tolist()}")
    
    print("-" * 80)
    print(f"**[최종 백도어 탐지 정확도 (Recall)]:** {detection_accuracy*100:.2f}%")
    print(f"**[Neural Cleanse 탐지 총 시간]:** {detection_time_ms:.4f} ms")
    print("="*80)
    
    return detection_accuracy, detection_time

# ------------------------------
# 메인 실행: 모델 학습 및 NC 탐지 수행
# ------------------------------
try:
    base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
except FileNotFoundError as e:
    print(f"\n[오류 발생] 데이터셋 경로를 찾을 수 없습니다: {e}")
    exit()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 학습 데이터: 50%의 백도어 샘플 포함 (공격 모델 생성)
train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.5, target_label=TARGET_LABEL, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# NC 평가용 데이터: 클린 이미지만 사용
test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
test_clean_loader = DataLoader(test_clean_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n[백도어 공격에 취약한 모델 학습 중... (5 에포크)]")
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)
print("\n[학습 완료]")

# ------------------------------
# Neural Cleanse 탐지 실행 및 최종 출력
# ------------------------------
# GTSRB는 43개 클래스를 가집니다.
nc_acc, nc_time = run_neural_cleanse(model, test_clean_loader, num_classes=43, device=device)

