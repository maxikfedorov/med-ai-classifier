import os
import pandas as pd

def load_folder_to_pathology_map(excel_file_path):
    df = pd.read_excel(excel_file_path)
    return dict(zip(df['study_instance_anon'], df['pathology']))

def find_dcm_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                return os.path.join(root, file)
    return None

def is_file_in_correct_folder(dcm_file_path, pathology):
    """
    Проверяет, находится ли файл dcm_file_path в правильной папке (0 или 1).
    """
    real_folder = os.path.basename(os.path.dirname(dcm_file_path))
    
    return str(pathology) == real_folder

def check_data_consistency(source_dir, excel_file_path, processed_dir):
    folder_to_pathology = load_folder_to_pathology_map(excel_file_path)
    errors = []
    
    for folder_name, pathology in folder_to_pathology.items():
        folder_path = os.path.join(source_dir, folder_name)
        dcm_file_path = find_dcm_file(folder_path)
        
        if not dcm_file_path:
            errors.append(f"DCM файл не найден в папке {folder_name}")
            continue
        
        processed_dcm_file_path = None
        for root, _, files in os.walk(processed_dir):
            if os.path.basename(dcm_file_path) in files:
                processed_dcm_file_path = os.path.join(root, os.path.basename(dcm_file_path))
                break
        
        if not processed_dcm_file_path:
            errors.append(f"Файл {os.path.basename(dcm_file_path)} отсутствует в обработанных данных")
            continue
        
        if not is_file_in_correct_folder(processed_dcm_file_path, pathology):
            errors.append(
                f"Ошибка: {processed_dcm_file_path} должен находиться в {processed_dir}\\{pathology}"
            )
    
    return errors

datasets = [
    {
        "source_dir": ("../data/ds_clav_fracture_train_good"),
        "excel_file": ("../data/ds_clav_fracture_train_good/block_0000_anon.xlsx"),
        "processed_dir": ("../data/raw_data/clav_fracture_dcm")
    },
    {
        "source_dir": ("../data/ds_medimp_train_good"),
        "excel_file": ("../data/ds_medimp_train_good/block_0000_anon.xlsx"),
        "processed_dir": ("../data/raw_data/medimp")
    }
]

for dataset in datasets:
    print(f"\nПроверка набора данных: {dataset['source_dir']}")
    errors = check_data_consistency(
        dataset["source_dir"],
        dataset["excel_file"],
        dataset["processed_dir"]
    )
    
    if errors:
        print("Обнаружены ошибки:")
        for error in errors:
            print(error)
    else:
        print("Все файлы находятся в правильных папках.")