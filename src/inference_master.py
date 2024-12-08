import os
import numpy as np
from tqdm import tqdm
import json
import inquirer
from tabulate import tabulate
from meta_model import EnsembleModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

RESULTS_DIR = "../results"

def get_true_label(image_name, dataset_name):
    raw_data_path = f"../data/raw_data/{dataset_name}_images"
    for class_dir in ['0', '1']:
        if os.path.exists(os.path.join(raw_data_path, class_dir, image_name)):
            return int(class_dir)
    return None

def save_results_to_json(results, accuracy, auc_score, dataset_name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    formatted_results = []
    for idx, result in enumerate(results, 1):
        formatted_results.append({
            "file_number": idx,
            "file_name": result["image_name"],
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "true_value": result["true_class"],
            "is_correct": result["is_correct"]
        })

    result_file_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    output_data = {
        "accuracy": f"{accuracy:.2f}%",
        "auc_roc": f"{auc_score:.4f}",
        "results": formatted_results
    }

    with open(result_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    print(f"\nРезультаты сохранены в файл: {result_file_path}")

def plot_roc_curve(y_true, y_pred, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC кривая (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC кривая для датасета {dataset_name}')
    plt.legend(loc="lower right")
    
    # Сохраняем график
    plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_roc_curve.png")
    plt.savefig(plot_path, dpi = 600)
    plt.close()
    
    return roc_auc

def select_dataset():
    datasets = ['clav_fracture', 'medimp']
    questions = [
        inquirer.List('dataset',
                     message="Выберите датасет для инференса:",
                     choices=datasets)
    ]
    answers = inquirer.prompt(questions)
    return answers['dataset']

def run_inference(dataset_name):
    ensemble = EnsembleModel(dataset_name)
    inference_dir = f'../data/processed_data/{dataset_name}/!user'
    
    if not os.path.exists(inference_dir):
        return {"error": "Директория с данными не найдена"}, 404
        
    image_files = [f for f in os.listdir(inference_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    correct_predictions = 0
    total_with_labels = 0
    
    y_true = []
    y_pred = []
    
    for image_file in image_files:
        image_path = os.path.join(inference_dir, image_file)
        prediction = ensemble.predict_single_image(image_path)
        
        true_label = get_true_label(image_file, dataset_name)
        pred_label = 1 if prediction[1] > 0.5 else 0
        
        if true_label is not None:
            total_with_labels += 1
            correct_predictions += int(pred_label == true_label)
            y_true.append(true_label)
            y_pred.append(prediction[1])
            
        results.append({
            "image_name": image_file,
            "predicted_class": pred_label,
            "confidence": float(prediction[1]),
            "true_class": true_label,
            "is_correct": pred_label == true_label if true_label is not None else None
        })
    
    accuracy = (correct_predictions / total_with_labels * 100) if total_with_labels > 0 else 0
    
    # Вычисляем AUC-ROC только если есть размеченные данные
    if y_true and y_pred:
        auc_score = plot_roc_curve(y_true, y_pred, dataset_name)
    else:
        auc_score = 0
        
    save_results_to_json(results, accuracy, auc_score, dataset_name)
    
    return {
        "accuracy": accuracy,
        "auc_roc": auc_score,
        "results": results,
        "total_images": len(image_files)
    }, 200

def main():
    dataset_name = select_dataset()
    ensemble = EnsembleModel(dataset_name)
    
    inference_dir = f'../data/processed_data/{dataset_name}/!user'
    image_files = [f for f in os.listdir(inference_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nНайдено {len(image_files)} изображений для инференса")
    
    results = []
    correct_predictions = 0
    total_with_labels = 0
    
    y_true = []
    y_pred = []
    
    for image_file in tqdm(image_files, desc="Обработка изображений"):
        image_path = os.path.join(inference_dir, image_file)
        prediction = ensemble.predict_single_image(image_path)
        
        true_label = get_true_label(image_file, dataset_name)
        pred_label = 1 if prediction[1] > 0.5 else 0
        
        if true_label is not None:
            total_with_labels += 1
            correct_predictions += int(pred_label == true_label)
            y_true.append(true_label)
            y_pred.append(prediction[1])
        
        results.append({
            "image_name": image_file,
            "predicted_class": pred_label,
            "confidence": float(prediction[1]),
            "true_class": true_label,
            "is_correct": "верно" if pred_label == true_label else "неверно" if true_label is not None else "N/A"
        })
    
    accuracy = (correct_predictions / total_with_labels * 100) if total_with_labels > 0 else 0
    
    # Вычисляем AUC-ROC только если есть размеченные данные
    if y_true and y_pred:
        auc_score = plot_roc_curve(y_true, y_pred, dataset_name)
    else:
        auc_score = 0
        
    save_results_to_json(results, accuracy, auc_score, dataset_name)
    
    table_data = [[
        r['image_name'][-15:],
        r['predicted_class'],
        f"{r['confidence']:.4f}",
        r['true_class'] if r['true_class'] is not None else 'N/A',
        r['is_correct']
    ] for r in results]
    
    headers = ['Изображение', 'Предикт', 'Уверенность', 'Истинное', 'Результат']
    print("\nРезультаты классификации:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nОбщая точность: {accuracy:.2f}%")
    print(f"AUC-ROC: {auc_score:.4f}")

if __name__ == "__main__":
    main()