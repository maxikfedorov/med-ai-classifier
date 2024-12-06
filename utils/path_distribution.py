import os
import shutil
import pandas as pd

data_dir = os.path.join('../data', 'ds_clav_fracture_train_good')
excel_file = None

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith('.xlsx') or file.lower().endswith('.xls'):
            excel_file = os.path.join(root, file)
            break
    if excel_file:
        break

if not excel_file:
    print("Excel файл не найден в указанной директории.")
    exit()

df = pd.read_excel(excel_file)

processed_dir = os.path.join('../data', 'processed', 'clav_fracture')
os.makedirs(os.path.join(processed_dir, '0'), exist_ok=True)
os.makedirs(os.path.join(processed_dir, '1'), exist_ok=True)

labels_dict = {str(row[1]).strip(): row[2] for row in df.itertuples(index=False)}

for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        if dir_name in labels_dict:
            label = labels_dict[dir_name]
            print(f"Обработка директории: {dir_name}, метка: {label}")
            
            dest_dir = os.path.join(processed_dir, str(label))
            
            current_dir = os.path.join(root, dir_name)
            
            for sub_root, sub_dirs, sub_files in os.walk(current_dir):
                for file in sub_files:
                    if file.lower().endswith('.dcm'):
                        src_file_path = os.path.join(sub_root, file)
                        shutil.copy(src_file_path, dest_dir)
                        print(f"Копирование файла: {file} в {dest_dir}")
        else:
            print(f"Метка для директории {dir_name} не найдена.")

print("Файлы успешно обработаны и распределены по папкам.")