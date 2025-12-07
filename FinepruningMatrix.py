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
import time # 속도 측정을 위해 time 라이브러리 추가
import copy # 모델 복사를 위해 copy 모듈 임포트
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Confusion Matrix 계산 및 시각화를 위해 추가
import matplotlib.pyplot as plt # 시각화를 위해 추가
import torch.nn.functional as F

# ------------------------------
# 상수 정의 및 실시간 방어 최적화 설정
# ------------------------------
# 경로 (실제 데이터셋이 없는 환경에서도 실행 가능하도록 더미 경로 설정)
# !!! 중요: 이 경로를 실제 GTSRB 데이터셋 경로로 변경해야 코드가 실행됩니다. !!!
IMG_DIR = r"경로 설정" 
LABEL_CSV = r"경로 설정"
TARGET_LABEL = 0 

# <<<< 실시간 방어 속도 최적화 설정 >>>>
BATCH_SIZE = 256 # 배치 사이즈를 높여 GPU 활용 및 추론 속도 개선
FP_FINETUNE_EPOCHS = 1 # 미세조정 에포크 수를 최소화하여 방어 시간 단축
NUM_CLASSES = 43 # GTSRB 클래스 수

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
        data = self.dataset[idx]
        if isinstance(data, tuple) and len(data) == 2:
            img, label = data
        else:
            raise ValueError(f"Base Dataset의 __getitem__이 예상치 않은 형식의 데이터를 반환했습니다: {data}")

        # Pytorch Tensor를 PIL Image로 변환 (트리거 삽입을 위해)
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            
        if random.random() < self.trigger_ratio:
            img = random.choice(trigger_pool)(img)
            label = self.target_label
            
        # 다시 Tensor로 변환
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
            img = self.transform(img)
        
        # 타겟 레이블로 고정
        label = torch.tensor(self.target_label).long()
        return img, label

class SimpleCNN(nn.Module):
    """간단한 2단 Convolutional Neural Network 모델"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1) # nn.Flatten()이 처리하므로 불필요
        x = self.classifier(x)
        return x

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
# Fine-Pruning 핵심 함수 
# ------------------------------
def get_avg_activations(model, dataloader, device):
    """모델의 특정 레이어 활성화 평균 측정"""
    activations = []
    hook_layer = model.features[3] # Pruning 대상 레이어 (Conv2d(32, 64, 3, padding=1))

    def hook_fn(module, input, output):
        # 배치, 높이, 너비에 걸쳐 채널별 평균 활성화 측정
        avg = output.mean(dim=(0, 2, 3)).detach().cpu()
        activations.append(avg)

    handle = hook_layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            model(imgs)
            if i >= 50: # 빠르게 샘플링하여 활성화 측정
                break

    handle.remove()
    act = torch.stack(activations).mean(dim=0)
    return act

def prune_model(model, threshold=0.05, device=None):
    """활성화 임계값을 기준으로 모델의 Conv2d 필터를 가지치기"""
    pruned_model = copy.deepcopy(model)
    with torch.no_grad():
        conv_layer = pruned_model.features[3]
        weight = conv_layer.weight.data
        bias = conv_layer.bias.data if conv_layer.bias is not None else None

        # 활성화 임계값 대신, 필터 가중치의 L1 노름 평균을 사용합니다.
        activation = weight.abs().mean(dim=(1, 2, 3)) 
        mask = activation > threshold
        kept_indices = mask.nonzero(as_tuple=True)[0]
        num_kept = len(kept_indices)
        
        if num_kept == 0:
            print("경고: 가지치기 후 남은 채널이 없어 강제로 1개 채널을 유지합니다.")
            num_kept = 1
            kept_indices = torch.tensor([0])
            
        print(f"가지치기 임계값 {threshold:.3f}, 원본 채널: 64개, 유지된 채널: {num_kept}개")

        # Conv2d 레이어 교체 (출력 채널 변경)
        new_conv = nn.Conv2d(32, num_kept, kernel_size=3, padding=1)
        new_conv.weight.data = weight[kept_indices]
        if bias is not None:
            new_conv.bias.data = bias[kept_indices]
        pruned_model.features[3] = new_conv.to(device)

        # Classifier 레이어 교체 (입력 크기 변경: num_kept * 8 * 8)
        pruned_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_kept * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        ).to(device)

    return pruned_model

def fine_tune(model, dataloader, device, epochs=FP_FINETUNE_EPOCHS):
    """가지치기된 모델을 미세조정(Fine-tuning)"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    print(f"미세조정 에포크: {epochs}")
    for epoch in range(epochs):
        train(model, dataloader, optimizer, criterion, device)

# ------------------------------
# Fine-Pruning 방어 후, 순수 추론 속도 및 정확도 측정 함수 (수정됨)
# ------------------------------
def measure_pruned_model_inference(model, loader, device):
    """
    방어된 모델 (가지치기된 모델)의 클린 데이터에 대한 정확도, 순수 추론 속도 및 
    2x2 오분류표를 측정하고 출력 및 시각화합니다.
    """
    model.eval()
    all_labels = []
    all_preds = []
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_samples = len(all_labels)
    
    # 1. 정확도 및 속도 계산
    correct = np.sum(np.array(all_labels) == np.array(all_preds))
    detection_accuracy = 100.0 * correct / total_samples
    time_per_sample = inference_time / total_samples if total_samples > 0 else 0.0
    
    # 2. 오분류표 (Confusion Matrix) 계산 (43x43)
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    
    # 타겟 클래스 (Class 0) 기준의 2x2 오분류표 지표 추출
    target_class = TARGET_LABEL
    TP_TARGET = cm[target_class, target_class] 
    FN_TARGET = np.sum(cm[target_class, :]) - TP_TARGET 
    FP_TARGET = np.sum(cm[:, target_class]) - TP_TARGET 
    TN_TARGET = total_samples - TP_TARGET - FN_TARGET - FP_TARGET 

    # --------------------------------------------------------------------------
    # 🌟 혼동 행렬 시각화 코드 (2x2 매트릭스: 타겟 0 vs 나머지) 🌟
    # --------------------------------------------------------------------------
    
    # 2x2 Confusion Matrix 구성
    cm_2x2 = np.array([
        [TP_TARGET, FN_TARGET], # 실제 0 (Target)
        [FP_TARGET, TN_TARGET]  # 실제 Other (Non-Target)
    ])

    # 시각화 (정수 count로 표시)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_2x2, 
        display_labels=[f'Class {TARGET_LABEL} (Target)', 'Other Classes']
    )
    disp.plot(cmap='Reds', values_format='d') # 방어 후 결과이므로 붉은 계열로 변경
    
    plt.title(f"Fine-Pruning 후 2x2 Confusion Matrix (Target Class {TARGET_LABEL} vs Others)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # --------------------------------------------------------------------------
    
    # 3. 결과 출력 (오분류표 형식)
    print("\n" + "="*70)
    print("                      *** Fine-Pruning 후 분류 성능 분석 ***")
    print(f"                      (클린 데이터셋, 타겟 클래스 {TARGET_LABEL} 기준 2x2)")
    print("-" * 70)
    print(f"       |      Predicted: Class {TARGET_LABEL} |       Predicted: Other      |")
    print("-" * 70)
    print(f"Actual |      {TP_TARGET:7d} (True Positive)    |       {FN_TARGET:7d} (False Negative)  |")
    print(f"Class {TARGET_LABEL}|")
    print("-" * 70)
    print(f"Actual |      {FP_TARGET:7d} (False Positive)   |       {TN_TARGET:7d} (True Negative)   |")
    print(f"Other |")
    print("-" * 70)
    
    # 4. 추론 속도 및 정확도 요약 출력
    print(f"\n[방어 후 전체 클린 정확도]: {detection_accuracy:.2f}%")
    print(f"[총 추론 시간]:              {inference_time:.4f} 초")
    print(f"[샘플당 추론 속도]:          {time_per_sample * 1000:.2f} ms/샘플 ({total_samples} 샘플 기준)")
    print("="*70)
    
    return detection_accuracy, inference_time

# ------------------------------
# 메인 실행 흐름
# ------------------------------
# 1. 데이터 로더 설정
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

try:
    base_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=None)
    # 백도어 학습을 위한 데이터셋 (50% 오염)
    train_dataset = GTSRBBackdoor(base_dataset, trigger_ratio=0.5, target_label=TARGET_LABEL, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 클린 테스트 데이터셋
    test_clean_dataset = CustomGTSRB(IMG_DIR, LABEL_CSV, transform=transform)
    test_clean_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 백도어 공격 성공률(ASR) 측정을 위한 데이터셋
    test_backdoor_dataset = TriggerOnlyDataset(test_clean_dataset, target_label=TARGET_LABEL, transform=transform)
    test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=BATCH_SIZE, shuffle=False)

except Exception as e:
    print(f"\n[오류 발생] 데이터셋 로드 중 문제가 발생했습니다: {e}")
    print("GTSRB 데이터셋 경로(IMG_DIR, LABEL_CSV)를 확인하거나, 더미 데이터 사용에 문제가 없는지 확인하십시오.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 2. 백도어 모델 학습 (Pruning 전)
print("\n[백도어 공격에 취약한 모델 학습 중...]")
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)
clean_acc_pre = evaluate(model, test_clean_loader, device)
asr_pre = evaluate(model, test_backdoor_loader, device)
print(f"[학습 완료] Clean Acc (Pruning 전): {clean_acc_pre:.2f}%, ASR (Pruning 전): {asr_pre:.2f}%")

# 3. Fine-Pruning 방어 기법 적용
print(f"\n[Fine-Pruning 방어 적용 중... (미세조정 {FP_FINETUNE_EPOCHS} 에포크)]")

defense_start_time = time.time() 

# a. 활성화 평균 측정
activations = get_avg_activations(model, train_loader, device) 

# b. 모델 가지치기 (Pruning)
# 활성화가 낮았던 (백도어 관련) 채널이 제거되어야 합니다.
pruned_model = prune_model(model, threshold=0.05, device=device)

# c. 가지치기된 모델 미세조정 (Fine-tuning)
fine_tune(pruned_model, train_loader, device, epochs=FP_FINETUNE_EPOCHS)

defense_end_time = time.time()
defense_time = defense_end_time - defense_start_time
print(f"\n**[총 방어 (Pruning + Fine-tuning) 소요 시간]:** {defense_time:.4f} 초")

# 4. 방어된 모델 (Pruned Model)의 순수 추론 속도 및 정확도 측정 (수정된 함수 호출)
measure_pruned_model_inference(pruned_model, test_clean_loader, device)

