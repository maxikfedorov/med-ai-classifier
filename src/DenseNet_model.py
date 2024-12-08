import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import inquirer
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from telegram_bot import send_notification, run_bot
import threading

# Оптимизированные гиперпараметры
BATCH_SIZE = 16
NUM_EPOCHS = 40 
LEARNING_RATE = 0.0001
IMG_SIZE = 224
BASE_DIR = "../data/processed_data"
CHECKPOINTS_DIR = "../checkpoints"
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-4

class ChestXRayDenseNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ChestXRayDenseNet, self).__init__()
        self.model_name = "densenet121"
        self.model = models.densenet121(pretrained=pretrained)
        
        # Заморозка начальных слоев
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
            
        # Модификация классификатора
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        return self.model(x)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def predict(self, img_tensor):
        self.eval()
        with torch.no_grad():
            output = self(img_tensor)
            probabilities = torch.softmax(output, dim=1)
        return probabilities

class ChestXRayTrainer:
    def __init__(self, model, device, dataset_name):
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        
        # Взвешенный лосс для несбалансированных классов
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # Оптимизатор с L2-регуляризацией
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Планировщик learning rate
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=SCHEDULER_PATIENCE,
            verbose=True
        )
        
    def create_dataloaders(self):
        train_dataset = ImageFolder(
            os.path.join(BASE_DIR, self.dataset_name, 'train'), 
            transform=self.model.transform
        )
        val_dataset = ImageFolder(
            os.path.join(BASE_DIR, self.dataset_name, 'val'), 
            transform=self.model.transform
        )
        test_dataset = ImageFolder(
            os.path.join(BASE_DIR, self.dataset_name, 'test'), 
            transform=self.model.transform
        )
        
        # Используем num_workers для ускорения загрузки данных
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Обучение')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        epoch_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        return epoch_loss, epoch_auc, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        return val_loss, val_auc, val_acc
    
    def train(self, num_epochs=NUM_EPOCHS):
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nЭпоха {epoch+1}/{num_epochs}')
            
            train_loss, train_auc, train_acc = self.train_epoch()
            val_loss, val_auc, val_acc = self.validate()
            
            # Обновление learning rate
            self.scheduler.step(val_auc)
            
            print(f'Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}')
            
            # Сохранение лучшей модели и early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                checkpoint_path = os.path.join(
                    CHECKPOINTS_DIR, 
                    f'best_densenet_{self.dataset_name}.pth'
                )
                self.model.save_weights(checkpoint_path)
                print(f'Сохранена лучшая модель с Val AUC: {val_auc:.4f}')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'\nОстановка обучения: нет улучшений в течение {EARLY_STOPPING_PATIENCE} эпох')
                break

def get_dataset_choice():
    datasets = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    questions = [
        inquirer.List('dataset',
                     message="Выберите датасет для обучения:",
                     choices=datasets)
    ]
    answers = inquirer.prompt(questions)
    return answers['dataset']

def main():
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = get_dataset_choice()
    
    # Инициализация модели и тренера
    model = ChestXRayDenseNet().to(device)
    trainer = ChestXRayTrainer(model, device, dataset_name)
    trainer.create_dataloaders()
    
    send_notification(f"Модель DenseNet начинает учиться на датасете {dataset_name}")
    
    # Запуск обучения
    trainer.train()
    
    send_notification(f"Модель DenseNet закончила учиться на датасете {dataset_name}")

if __name__ == '__main__':
    main()