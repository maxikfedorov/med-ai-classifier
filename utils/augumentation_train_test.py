import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random

def create_directories(base_path):
    splits = ['train', 'val', 'test']
    classes = ['0', '1']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)

def add_noise(image):
    # Преобразуем изображение в float32
    image_float = image.astype(np.float32)
    noise = np.random.normal(0, 15, image.shape).astype(np.float32)
    # Складываем и обрезаем значения до диапазона [0, 255]
    noisy_image = np.clip(image_float + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_scratches(image):
    scratch = image.copy()
    for _ in range(random.randint(3, 7)):
        pt1 = (random.randint(0, 223), random.randint(0, 223))
        pt2 = (random.randint(0, 223), random.randint(0, 223))
        cv2.line(scratch, pt1, pt2, (255, 255, 255), 1)
    return scratch

def augment_image(image):
    image = cv2.resize(image, (224, 224))
    
    augmentations = []

    for flip in [True, False]:
        for rotation in [0, -10, 10, -15, 15]:
            for blur_kernel in [(0, 0), (3, 3), (7, 7), (9, 9)]:
                for noise in [True, False]:
                    for scratch in [True, False]:
                        img_aug = image.copy()
                        
                        if flip:
                            img_aug = cv2.flip(img_aug, 1)
                            
                        if rotation != 0:
                            matrix = cv2.getRotationMatrix2D((112, 112), rotation, 1.0)
                            img_aug = cv2.warpAffine(img_aug, matrix, (224, 224))
                            
                        if blur_kernel != (0, 0):
                            img_aug = cv2.GaussianBlur(img_aug, blur_kernel, 0)
                            
                        if noise:
                            img_aug = add_noise(img_aug)
                            
                        if scratch:
                            img_aug = add_scratches(img_aug)
                            
                        augmentations.append(img_aug)
    
    return augmentations

def process_images():
    input_base = '../data/raw_data/medimp_images'
    output_base = '../data/processed_data/medimp'
    
    create_directories(output_base)
    
    for class_name in ['0', '1']:
        input_dir = os.path.join(input_base, class_name)
        images = os.listdir(input_dir)
        
        # Разделение на train/val/test
        train_imgs, temp_imgs = train_test_split(images, train_size=0.7, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=0.67, random_state=42)
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, split_images in splits.items():
            for img_name in split_images:
                img_path = os.path.join(input_dir, img_name)
                image = cv2.imread(img_path)
                
                augmented_images = augment_image(image)
                
                for i, aug_img in enumerate(augmented_images):
                    output_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    output_path = os.path.join(output_base, split_name, class_name, output_name)
                    cv2.imwrite(output_path, aug_img)

if __name__ == "__main__":
    process_images()