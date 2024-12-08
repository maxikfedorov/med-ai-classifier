import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
import inquirer
from tabulate import tabulate
from DenseNet_model import ChestXRayDenseNet
from ResNet_model import get_model as get_resnet
from VGG_model import VGGClassifier

RESULTS_DIR = "../results"

class EnsembleModel:
    def __init__(self, dataset_name, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        
        # Инициализация моделей
        self.densenet = ChestXRayDenseNet().to(self.device)
        self.resnet = get_resnet().to(self.device)
        self.vgg = VGGClassifier(num_classes=2).to(self.device)
        
        # Загрузка весов
        self.densenet.load_weights(f'../checkpoints/best_densenet_{dataset_name}.pth')
        self.resnet.load_state_dict(torch.load(f'../checkpoints/best_model_{dataset_name}.pth')['model_state_dict'])
        self.vgg.load_state_dict(torch.load(f'../checkpoints/best_vgg_{dataset_name}.pth'))
        
        for model in [self.densenet, self.resnet, self.vgg]:
            model.eval()
        
        # Трансформации
        self.transforms = {
            'densenet': self.densenet.transform,
            'resnet': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'vgg': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def predict_single_image(self, image_path, weights=[0.4, 0.3, 0.3]):
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            densenet_input = self.transforms['densenet'](image).unsqueeze(0).to(self.device)
            resnet_input = self.transforms['resnet'](image).unsqueeze(0).to(self.device)
            vgg_input = self.transforms['vgg'](image).unsqueeze(0).to(self.device)
            
            densenet_output = torch.softmax(self.densenet(densenet_input), dim=1)
            resnet_output = torch.softmax(self.resnet(resnet_input), dim=1)
            vgg_output = torch.softmax(self.vgg(vgg_input), dim=1)
            
            ensemble_pred = (
                weights[0] * densenet_output.cpu().numpy() +
                weights[1] * resnet_output.cpu().numpy() +
                weights[2] * vgg_output.cpu().numpy()
            )
            
        return ensemble_pred[0]

def get_true_label(image_name, dataset_name):
    raw_data_path = f"../data/raw_data/{dataset_name}_images"
    for class_dir in ['0', '1']:
        if os.path.exists(os.path.join(raw_data_path, class_dir, image_name)):
            return int(class_dir)
    return None

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

def select_dataset():
    datasets = ['clav_fracture', 'medimp']
    questions = [
        inquirer.List('dataset',
                     message="Выберите датасет для инференса:",
                     choices=datasets)
    ]
    answers = inquirer.prompt(questions)
    return answers['dataset']

def main():
    dataset_name = select_dataset()
    
    # Инициализация ансамбля
    ensemble = EnsembleModel(dataset_name)
    
    inference_dir = f'../data/processed_data/{dataset_name}/!user'
    image_files = [f for f in os.listdir(inference_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nНайдено {len(image_files)} изображений для инференса")
    
    results = []
    correct_predictions = 0
    total_with_labels = 0
    
    for image_file in tqdm(image_files, desc="Обработка изображений"):
        image_path = os.path.join(inference_dir, image_file)
        prediction = ensemble.predict_single_image(image_path)
        
        true_label = get_true_label(image_file, dataset_name)
        pred_label = 1 if prediction[1] > 0.5 else 0
        
        if true_label is not None:
            total_with_labels += 1
            correct_predictions += int(pred_label == true_label)
        
        results.append({
            "image_name": image_file,
            "predicted_class": pred_label,
            "confidence": float(prediction[1]),
            "true_class": true_label,
            "is_correct": "верно" if pred_label == true_label else "неверно" if true_label is not None else "N/A"
        })
    
    # Вычисление точности
    accuracy = (correct_predictions / total_with_labels * 100) if total_with_labels > 0 else 0
    
    # Сохранение результатов в JSON
    save_results_to_json(results, accuracy, dataset_name)
    
    # Подготовка данных для таблицы
    table_data = []
    for r in results:
        table_data.append([
            r['image_name'][-15:],
            r['predicted_class'],
            f"{r['confidence']:.4f}",
            r['true_class'] if r['true_class'] is not None else 'N/A',
            r['is_correct']
        ])
    
    # Вывод таблицы
    headers = ['Изображение', 'Предикт', 'Уверенность', 'Истинное', 'Результат']
    print("\nРезультаты классификации:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nОбщая точность: {accuracy:.2f}%")

if __name__ == "__main__":
    main()