import os
import torch
from pathlib import Path
import inquirer
from tabulate import tabulate
from PIL import Image
import numpy as np
from DenseNet_model import ChestXRayDenseNet

# Константы
BASE_DIR = "../data/processed_data"
RAW_DIR = "../data/raw_data"
CHECKPOINTS_DIR = "../checkpoints"

def get_dataset_choice():
    datasets = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    questions = [
        inquirer.List('dataset',
                     message="Выберите датасет для инференса:",
                     choices=datasets)
    ]
    answers = inquirer.prompt(questions)
    return answers['dataset']

def get_true_label(image_name, dataset_name):
    """Поиск истинного класса изображения в raw_data"""
    raw_path = Path(RAW_DIR) / f"{dataset_name}_images"
    
    # Рекурсивный поиск файла
    for path in raw_path.rglob(image_name):
        # Определяем класс по родительской папке
        return 1 if path.parent.name == '1' else 0
    
    return None

def load_model(dataset_name, device):
    """Загрузка модели с весами для выбранного датасета"""
    model = ChestXRayDenseNet().to(device)
    weights_path = os.path.join(CHECKPOINTS_DIR, f'best_densenet_{dataset_name}.pth')
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Загружены веса модели: {weights_path}")
    else:
        raise FileNotFoundError(f"Не найдены веса модели для датасета {dataset_name}")
    
    return model

def predict_images(model, dataset_name):
    """Предсказание для всех изображений в пользовательской папке"""
    device = next(model.parameters()).device
    user_dir = os.path.join(BASE_DIR, dataset_name, '!user')
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    if not os.path.exists(user_dir):
        print(f"Папка {user_dir} не найдена!")
        return
    
    for image_name in os.listdir(user_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(user_dir, image_name)
            
            # Загрузка и преобразование изображения
            image = Image.open(image_path).convert('RGB')
            image_tensor = model.transform(image).unsqueeze(0).to(device)
            
            # Получение предсказания
            with torch.no_grad():
                probabilities = model.predict(image_tensor)
                pred_prob = probabilities[0][1].item()
                pred_class = 1 if pred_prob >= 0.5 else 0
            
            # Получение истинного класса
            true_class = get_true_label(image_name, dataset_name)
            
            # Проверка корректности предсказания
            is_correct = "Верно" if pred_class == true_class else "Неверно"
            if pred_class == true_class:
                correct_predictions += 1
            total_predictions += 1
            
            # Сокращение имени файла до последних 15 символов
            short_name = image_name[-15:] if len(image_name) > 15 else image_name
            
            results.append([
                short_name,
                pred_class,
                f"{pred_prob:.3f}",
                true_class,
                is_correct
            ])
    
    # Вывод результатов в виде таблицы
    headers = ["Изображение", "Предикт", "Уверенность", "Истинный", "Результат"]
    print("\nРезультаты предсказаний:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Вывод общей точности
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nОбщая точность: {accuracy:.2%}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Выбор датасета
    dataset_name = get_dataset_choice()
    
    try:
        # Загрузка модели
        model = load_model(dataset_name, device)
        model.eval()
        
        # Запуск предсказаний
        predict_images(model, dataset_name)
        
    except Exception as e:
        print(f"Ошибка при выполнении инференса: {str(e)}")

if __name__ == "__main__":
    main()