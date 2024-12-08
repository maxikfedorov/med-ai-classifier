import warnings
warnings.filterwarnings('ignore')

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import inquirer
from tabulate import tabulate  # Для вывода таблицы

# =======================
# Configurable Parameters
# =======================
DATA_PATH = "../data/processed_data"
RAW_DATA_PATH = "../data/raw_data"
CHECKPOINT_DIR = "../checkpoints"
RESULTS_DIR = "../results"
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
# Model Definition
# =======================
class VGGClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

# =======================
# Dataset Selection
# =======================
def select_dataset(data_path):
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    questions = [inquirer.List('dataset', message="Выберите датасет", choices=datasets)]
    answers = inquirer.prompt(questions)
    return answers['dataset']

# =======================
# Load Model
# =======================
def load_model(dataset_name):
    model_path = os.path.join(CHECKPOINT_DIR, f"best_vgg_{dataset_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель для датасета '{dataset_name}' не найдена: {model_path}")

    model = VGGClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Модель для датасета '{dataset_name}' успешно загружена.")
    return model

# =======================
# Get True Labels
# =======================
def get_true_label(image_name, raw_data_folder):
    for label in ['0', '1']:
        label_folder = os.path.join(raw_data_folder, label)
        if image_name in os.listdir(label_folder):
            return int(label)
    raise ValueError(f"Истинное значение для изображения '{image_name}' не найдено в папках '0' или '1'.")

# =======================
# Image Classification
# =======================
def classify_images(model, image_folder, raw_data_folder):
    results = []

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {image_folder}")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"В папке '{image_folder}' нет изображений для классификации.")

    print(f"Классификация изображений из папки: {image_folder}")
    
    correct_predictions = 0

    for idx, image_file in enumerate(image_files, start=1):  # Добавлен порядковый номер (idx)
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            confidence, prediction = torch.max(probabilities, dim=0)

        true_label = get_true_label(image_file, raw_data_folder)
        is_correct = prediction.item() == true_label
        correct_predictions += int(is_correct)

        results.append({
            "Номер": idx,
            "Название файла": image_file,
            "Предсказание": "Патология" if prediction.item() == 1 else "Норма",
            "Уверенность": f"{confidence.item():.2f}",
            "Истинное значение": "Патология" if true_label == 1 else "Норма",
            "Вывод": "Верно" if is_correct else "Неверно"
        })

    accuracy = correct_predictions / len(image_files) * 100
    return results, accuracy

# =======================
# Save Results to JSON
# =======================
def save_results_to_json(results, accuracy, dataset_name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    result_file_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    
    output_data = {
        "accuracy": f"{accuracy:.2f}%",
        "results": results
    }
    
    with open(result_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"\nРезультаты сохранены в файл: {result_file_path}")

# =======================
# Main Function
# =======================

def main(dataset_name=None, interactive_mode=True):
    """
        interactive_mode (bool): Флаг интерактивного режима
    """
    if interactive_mode:
        dataset_name = select_dataset(DATA_PATH)
    elif not dataset_name:
        raise ValueError("В неинтерактивном режиме необходимо указать dataset_name")
    
    user_image_folder = os.path.join(DATA_PATH, dataset_name, "!user")
    raw_data_folder = os.path.join(RAW_DATA_PATH, f"{dataset_name}_images")
    
    model = load_model(dataset_name)
    results, accuracy = classify_images(model, user_image_folder, raw_data_folder)
    
    if interactive_mode:
        # Вывод в консоль только в интерактивном режиме
        print("\nРезультаты классификации:")
        table_data = [[res["Номер"], res["Название файла"][-10:], res["Предсказание"], 
                      res["Уверенность"], res["Истинное значение"], res["Вывод"]] for res in results]
        print(tabulate(table_data, headers=["№", "Название файла", "Предсказание", 
                      "Уверенность", "Истинное значение", "Вывод"], tablefmt="grid"))
        print(f"\nТочность классификации: {accuracy:.2f}% ({len([r for r in results if r['Вывод'] == 'Верно'])}/{len(results)})")
    
    save_results_to_json(results, accuracy, dataset_name)
    
    return {
        "accuracy": f"{accuracy:.2f}%",
        "results": results
    }

if __name__ == "__main__":
    main(interactive_mode=True)
