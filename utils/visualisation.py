import os
import pydicom
from PIL import Image
import numpy as np

def convert_dicom_to_jpeg(dicom_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                
                ds = pydicom.dcmread(dicom_path)
                
                image_array = ds.pixel_array
                image_8bit = (image_array / np.max(image_array) * 255).astype(np.uint8)
                img = Image.fromarray(image_8bit)
                
                relative_path = os.path.relpath(root, dicom_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                jpeg_filename = os.path.splitext(file)[0] + '.jpg'
                output_path = os.path.join(output_subdir, jpeg_filename)
                
                img.save(output_path, 'JPEG')
                print(f"Сохранено изображение: {output_path}")

clav_fracture_dicom_dir = os.path.join('data', 'raw_data', 'clav_fracture')
clav_fracture_output_dir = os.path.join('data', 'raw_data', 'clav_fracture_images')

medimp_dicom_dir = os.path.join('data', 'raw_data', 'medimp')
medimp_output_dir = os.path.join('data', 'raw_data', 'medimp_images')

convert_dicom_to_jpeg(clav_fracture_dicom_dir, clav_fracture_output_dir)
convert_dicom_to_jpeg(medimp_dicom_dir, medimp_output_dir)

print("Все файлы успешно конвертированы.")