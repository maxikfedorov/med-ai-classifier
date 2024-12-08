import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from DenseNet_model import ChestXRayDenseNet
from ResNet_model import get_model as get_resnet
from VGG_model import VGGClassifier

# Веса для ансамбля моделей
ENSEMBLE_WEIGHTS = {
    'densenet': 0.3,
    'resnet': 0.4,
    'vgg': 0.3
}

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

    def predict_single_image(self, image_path, weights=[0.4, 0.3, 0.5]):
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            densenet_input = self.transforms['densenet'](image).unsqueeze(0).to(self.device)
            resnet_input = self.transforms['resnet'](image).unsqueeze(0).to(self.device)
            vgg_input = self.transforms['vgg'](image).unsqueeze(0).to(self.device)
            
            densenet_output = torch.softmax(self.densenet(densenet_input), dim=1)
            resnet_output = torch.softmax(self.resnet(resnet_input), dim=1)
            vgg_output = torch.softmax(self.vgg(vgg_input), dim=1)
            
            ensemble_pred = (
                ENSEMBLE_WEIGHTS['densenet'] * densenet_output.cpu().numpy() +
                ENSEMBLE_WEIGHTS['resnet'] * resnet_output.cpu().numpy() +
                ENSEMBLE_WEIGHTS['vgg'] * vgg_output.cpu().numpy()
            )
            
        return ensemble_pred[0]