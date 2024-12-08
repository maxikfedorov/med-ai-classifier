import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score

from telegram_bot import send_notification, run_bot
import threading

# Гиперпараметры
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = '../checkpoints'    

DATASETS = {
    'clav_fracture': '../data/processed_data/clav_fracture',
    'medimp': '../data/processed_data/medimp'
}

class ClavicleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        for class_id in ['0', '1']:
            class_dir = os.path.join(data_dir, class_id)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(int(class_id))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_model(num_classes=2):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256), 
        nn.Dropout(0.7),     
        nn.Linear(256, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def print_and_log(message):
    print(f"[INFO] {message}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    all_labels = []
    all_probs = []
    
    epoch_start = time.time()
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        running_loss += loss.item()
        running_acc += calculate_metrics(outputs, labels)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, epoch_auc, epoch_time

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += loss.item()
            running_acc += calculate_metrics(outputs, labels)
    
    avg_loss = running_loss / len(data_loader)
    avg_acc = running_acc / len(data_loader)
    avg_auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, avg_acc, avg_auc

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def main(dataset_name='clav_fracture'):
    print_and_log("Начало выполнения скрипта")
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    send_notification(f"Модель ResNet начинает учиться на датасете {dataset_name}")
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Набор данных {dataset_name} не найден. Доступные: {list(DATASETS.keys())}")
    
    data_dir = DATASETS[dataset_name]
    checkpoint_file = f"best_model_{dataset_name}.pth"

    torch.manual_seed(42)
    np.random.seed(42)
    
    print_and_log(f"Используемое устройство: {DEVICE}")
    print_and_log(f"Параметры обучения: Batch Size={BATCH_SIZE}, Epochs={NUM_EPOCHS}, LR={LEARNING_RATE}")
    print_and_log(f"Выбранный набор данных: {dataset_name}")

    transform = get_transforms()

    train_dataset = ClavicleDataset(os.path.join(data_dir, 'train'), transform)
    val_dataset = ClavicleDataset(os.path.join(data_dir, 'val'), transform)
    test_dataset = ClavicleDataset(os.path.join(data_dir, 'test'), transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = get_model()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print_and_log(f"\nЭпоха {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc, train_auc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print_and_log(f"Время эпохи: {epoch_time:.2f} сек")
        print_and_log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print_and_log(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, checkpoint_path)
            
            print_and_log(f"Сохранена новая лучшая модель! Точность: {best_val_acc:.4f}")
            
    send_notification(f"Модель ResNet закончила учиться на датасете {dataset_name}")
    print_and_log("\nОбучение завершено!")
    print_and_log(f"Лучшая точность на валидации: {best_val_acc:.4f} (Эпоха {best_epoch})")

    # Оценка на тестовом наборе
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, DEVICE)
    print_and_log(f"\nРезультаты на тестовом наборе:")
    print_and_log(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    available_datasets = ['clav_fracture', 'medimp']
    
    print("Доступные наборы данных:")
    for i, dataset in enumerate(available_datasets):
        print(f"{i + 1}. {dataset}")
    
    try:
        choice = int(input("\nВведите номер набора данных для обучения (например, 1 или 2): "))
        if choice < 1 or choice > len(available_datasets):
            raise ValueError("Неверный выбор.")
        
        dataset_name = available_datasets[choice - 1]
        print(f"\nВы выбрали: {dataset_name}")

        main(dataset_name=dataset_name)
    
    except ValueError as e:
        print(f"Ошибка: {e}. Пожалуйста, введите корректный номер из списка.")
