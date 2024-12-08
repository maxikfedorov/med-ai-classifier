# Классификатор медицинских рентгенограмм

Проект представляет собой систему классификации медицинских рентгеновских снимков с использованием различных архитектур нейронных сетей.

## Требования

- Python 3.8+
- CUDA-совместимая видеокарта
- Node.js и npm для веб-интерфейса

## Установка

1. Клонируйте репозиторий и создайте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Установите зависимости для веб-интерфейса:

```bash
cd web/interface-app
npm install
```

## Структура проекта

Проект организован следующим образом:

- `src/` - исходный код моделей и инференса
- `checkpoints/` - сохранённые веса моделей
- `data/` - наборы данных
- `web/` - веб-приложение (Flask + Node.js)
- `utils/` - вспомогательные скрипты

```
📁 med-ai-classifier
├── 📄 .env
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 Код проекта HACK-CDIT.txt
│
├── 📁 .legacy
│   ├── 📄 densnet_inference.py
│   ├── 📄 inferense_master.py
│   ├── 📄 resnet_inference.py
│   └── 📄 vgg_inference.py
│
├── 📁 checkpoints
│   ├── 📄 best_densenet_clav_fracture.pth
│   ├── 📄 best_densenet_medimp.pth
│   ├── 📄 best_model_clav_fracture.pth
│   ├── 📄 best_model_medimp.pth
│   ├── 📄 best_vgg_clav_fracture.pth
│   └── 📄 best_vgg_medimp.pth
│
├── 📁 data
│   ├── 📁 processed_data
│   │   ├── 📁 clav_fracture
│   │   │   ├── 📁 !user
│   │   │   ├── 📁 test
│   │   │   ├── 📁 train
│   │   │   └── 📁 val
│   │   └── 📁 medimp
│   │       ├── 📁 !user
│   │       ├── 📁 test
│   │       ├── 📁 train
│   │       └── 📁 val
│   └── 📁 raw_data
│       ├── 📁 clav_fracture_dcm
│       ├── 📁 clav_fracture_images
│       ├── 📁 medimp_dcm
│       └── 📁 medimp_images
│
├── 📁 results
│   ├── 📄 clav_fracture_results.json
│   ├── 📄 clav_fracture_roc_curve.png
│   ├── 📄 medimp_results.json
│   └── 📄 medimp_roc_curve.png
│
├── 📁 src
│   ├── 📄 DenseNet_model.py
│   ├── 📄 inference_master.py
│   ├── 📄 meta_model.py
│   ├── 📄 ResNet_model.py
│   ├── 📄 telegram_bot.py
│   ├── 📄 VGG_model.py
│   └── 📄 __init__.py
│
├── 📁 utils
│   ├── 📄 augumentation_train_test.py
│   ├── 📄 info.py
│   ├── 📄 path_distribution.py
│   └── 📄 visualisation.py
│
└── 📁 web
    ├── 📄 app.py
    ├── 📄 routes.py
    ├── 📄 utils.py
    └── 📁 interface-app
        ├── 📄 package.json
        ├── 📁 public
        │   ├── 📄 index.html
        │   ├── 📁 css
        │   └── 📁 js
        └── 📁 server
            └── 📄 server.js
```

## Обучение моделей

Для обучения моделей выполните следующие команды из директории `src`:

```bash
python VGG_model.py
python ResNet_model.py
python DenseNet_model.py
```

Веса лучших моделей автоматически сохраняются в директории `checkpoints/` с префиксом `best_`:
  - `best_vgg_clav_fracture.pth` - VGG модель для классификации переломов ключицы
  - `best_vgg_medimp.pth` - VGG модель для классификации медицинских имплантов
  - `best_densenet_clav_fracture.pth` - DenseNet модель для переломов
  - `best_densenet_medimp.pth` - DenseNet модель для имплантов
  - `best_model_clav_fracture.pth` - ResNet модель для переломов
  - `best_model_medimp.pth` - ResNet модель для имплантов

## Запуск веб-приложения

1. Запустите Flask-сервер:

```bash
cd web
python app.py
```

2. В отдельном терминале запустите фронтенд:

```bash
cd web/interface-app/server
npm start
```

## Доступ к приложению

- Backend API: http://localhost:5000
- Веб-интерфейс: http://localhost:3000

## Инференс

Для запуска инференса на новых данных используйте:

```bash
cd src
python inference_master.py
```

Результаты инференса будут сохранены в директории `results/`:
- JSON-файлы с метриками
- ROC-кривые в формате PNG

## Структура данных

- `processed_data/` - обработанные наборы данных
  - `clav_fracture/` - данные по переломам ключицы
  - `medimp/` - данные по медицинским имплантам

- `raw_data/` - исходные наборы данных
  - Содержит DICOM-файлы и преобразованные изображения
  - Разделение по классам (0/1)

**Исходный код проекта доступен в репозитории:**

> 🔗 [github.com/maxikfedorov/med-ai-classifier](https://github.com/maxikfedorov/med-ai-classifier)