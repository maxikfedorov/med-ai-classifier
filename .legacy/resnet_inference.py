import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import sys
import torch
from torch import nn
import time
from PIL import Image
import inquirer
from ResNet_model import get_model, get_transforms

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Конфигурация параметров инференса."""
    data_dir: Path
    raw_data_dir: Path
    model_path: Path
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_filename_length: int = 17
    truncated_length: int = 14

    CONFIGS = {
        'clav': {
            'data_dir': Path("../data/processed_data/clav_fracture/!user/"),
            'raw_data_dir': Path("../data/raw_data/clav_fracture_images"),
            'model_path': Path("../checkpoints/best_model_clav_fracture.pth"),
        },
        'medimp': {
            'data_dir': Path("../data/processed_data/medimp/!user"),
            'raw_data_dir': Path("../data/raw_data/medimp_images"),
            'model_path': Path("../checkpoints/best_model_medimp.pth"),
        },
    }

    @classmethod
    def create_config(cls, dataset_type: str) -> 'Config':
        """Создание конфигурации на основе выбранного набора данных."""
        try:
            config_params = cls.CONFIGS[dataset_type]
        except KeyError:
            raise ValueError(f"Неподдерживаемый тип датасета: {dataset_type}")
        return cls(**config_params)

class PredictionResult:
    """Класс для хранения результатов предсказания."""
    def __init__(self, predicted_class: Optional[int], confidence: Optional[float], true_label: Optional[int]):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.true_label = true_label
        
    @property
    def is_valid(self) -> bool:
        return all(x is not None for x in [self.predicted_class, self.confidence, self.true_label])
    
    @property
    def is_correct(self) -> bool:
        return self.predicted_class == self.true_label if self.is_valid else False
    
    def get_prediction_text(self) -> str:
        return "Патология" if self.predicted_class == 1 else "Норма"
    
    def get_true_label_text(self) -> str:
        return "Патология" if self.true_label == 1 else "Норма"

class ModelInference:
    """Класс для выполнения инференса модели."""
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.transform = get_transforms()
        
    def _load_model(self) -> nn.Module:
        """Загрузка и подготовка модели."""
        model = get_model()
        checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.config.device)
        model.eval()
        return model
    
    def get_true_label(self, image_name: str) -> Optional[int]:
        """Определение истинной метки изображения."""
        for label in [0, 1]:
            if (self.config.raw_data_dir / str(label) / image_name).exists():
                return label
        return None
    
    def predict_image(self, image_path: Path) -> Optional[Tuple[int, float]]:
        """Получение предсказания для одного изображения."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_path}: {str(e)}")
            return None
    
    def process_images(self) -> List[PredictionResult]:
        """Обработка всех изображений и получение результатов."""
        results = []
        image_paths = list(self.config.data_dir.glob("*"))
        
        for image_path in image_paths:
            prediction = self.predict_image(image_path)
            if prediction:
                predicted_class, confidence = prediction
                true_label = self.get_true_label(image_path.name)
                results.append(PredictionResult(predicted_class, confidence, true_label))
            else:
                results.append(PredictionResult(None, None, None))
                
        return results

class ResultsPrinter:
    """Класс для вывода результатов."""
    @staticmethod
    def print_header():
        header = "{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            "Изображение", "Предсказание", "Истинный класс", "Уверенность", "Результат"
        )
        logger.info("\n" + header)
        logger.info("-" * 80)
    
    @staticmethod
    def print_result(filename: str, result: PredictionResult):
        if result.is_valid:
            print("{:<20} {:<15} {:<15} {:.2%}         {:<15}".format(
                filename[:14] + "..." if len(filename) > 17 else filename,
                result.get_prediction_text(),
                result.get_true_label_text(),
                result.confidence,
                "Верно" if result.is_correct else "Неверно"
            ))
        else:
            print("{:<20} {:<15}".format(filename, "Ошибка обработки"))
    
    @staticmethod
    def print_summary(results: List[PredictionResult]):
        valid_results = [r for r in results if r.is_valid]
        correct_predictions = sum(1 for r in valid_results if r.is_correct)
        total_predictions = len(valid_results)
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info("\n" + "-" * 80)
            logger.info(f"\nИтоговая точность: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

def main():
    while True:
        # Выбор датасета через inquirer
        dataset_question = [
            inquirer.List(
                'dataset',
                message="Выберите набор данных",
                choices=['Переломы ключицы (clav)', 'Медицинские импланты (medimp)', 'Выход'],
            )
        ]
        
        dataset_choice = inquirer.prompt(dataset_question)['dataset']
        
        if dataset_choice == 'Выход':
            print("Программа завершена.")
            break
        
        dataset_type = 'clav' if 'ключицы' in dataset_choice else 'medimp'
        
        # Создание конфигурации на основе выбора пользователя
        try:
            config = Config.create_config(dataset_type)
            
            if not config.data_dir.exists():
                logger.error(f"Директория {config.data_dir} не найдена")
                continue
            
            if not config.model_path.exists():
                logger.error(f"Модель {config.model_path} не найдена")
                continue
            
            logger.info(f"Используемое устройство: {config.device}")
            
            inference = ModelInference(config)
            results = inference.process_images()

            printer = ResultsPrinter()
            printer.print_header()
            
            for image_path, result in zip(config.data_dir.glob("*"), results):
                printer.print_result(image_path.name, result)
            
            printer.print_summary(results)
            
            logger.info("\nКлассификация завершена!")
            
            time.sleep(1.5)  
        
        except Exception as e:
            logger.error(f"Ошибка выполнения: {e}")

if __name__ == "__main__":
    main()