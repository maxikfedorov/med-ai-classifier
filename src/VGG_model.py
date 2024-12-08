import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score
import inquirer
from tqdm import tqdm

from telegram_bot import send_notification, run_bot
import threading

# =======================
# Configurable Parameters
# =======================
DATA_PATH = "../data/processed_data"
CHECKPOINT_DIR = "../checkpoints"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Data Transformation
# =======================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =======================
# Dataset Selection
# =======================
def select_dataset(data_path):
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    questions = [inquirer.List('dataset', message="Select a dataset", choices=datasets)]
    answers = inquirer.prompt(questions)
    return answers['dataset']

# =======================
# Model Definition
# =======================
class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)

# =======================
# Training Function
# =======================
def train_model(model, dataloaders, criterion, optimizer, num_epochs, checkpoint_path, patience):
    print("Начало обучения...")
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    counter = 0    # счетчик эпох без улучшения
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            with tqdm(total=len(dataloaders[phase]), desc=f"{phase.capitalize()} Progress") as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

                    pbar.update(1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc_roc = roc_auc_score(all_labels, all_preds) if phase == 'val' else None

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}", end="")
            if phase == 'val':
                print(f" AUC-ROC: {epoch_auc_roc:.4f}")
                
                # Проверка для early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    counter = 0
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Лучшая модель сохранена в {checkpoint_path}")
                else:
                    counter += 1
            else:
                print()

        # Early stopping
        if counter >= patience:
            print(f"\nРаннее прекращение обучения! Нет улучшения в течение {patience} эпох")
            break

    print("\nОбучение завершено!")
    print(f"Лучшее качество на валидации: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# =======================
# Main Function
# =======================
def main():
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
       
    dataset_name = select_dataset(DATA_PATH)
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    
    data_loaders = {
        x: DataLoader(
            ImageFolder(os.path.join(dataset_path, x), transform),
            batch_size=BATCH_SIZE,
            shuffle=(x == 'train')
        ) for x in ['train', 'val']
    }

    model = VGGClassifier(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_vgg_{dataset_name}.pth")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    send_notification(f"Модель VGG начинает учиться на датасете {dataset_name}")

    train_model(
        model, 
        data_loaders, 
        criterion, 
        optimizer, 
        NUM_EPOCHS, 
        checkpoint_path,
        EARLY_STOPPING_PATIENCE
    )
    
    send_notification(f"Модель VGG закончила учиться на датасете {dataset_name}")

if __name__ == "__main__":
    main()