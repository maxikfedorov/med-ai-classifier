from flask import jsonify, request
import os
import sys
import json
from utils import allowed_file, custom_secure_filename, get_upload_folder

# Константы
ALLOWED_DATASETS = ['medimp', 'clav_fracture']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Добавляем пути в sys.path
sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)
from inference_master import run_inference

def register_routes(app):
    
    @app.route('/upload/<dataset_name>', methods=['POST'])
    def upload_files(dataset_name):
        """
        Маршрут для загрузки файлов пользователем.
        """
        if dataset_name not in ALLOWED_DATASETS:
            return jsonify({
                'status': 'error',
                'message': f'Неверное имя датасета. Допустимые значения: {", ".join(ALLOWED_DATASETS)}'
            }), 400

        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Файл не найден в запросе'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Файл не выбран'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'Недопустимый формат файла. Разрешенные форматы: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        filename = custom_secure_filename(file.filename)
        upload_folder = get_upload_folder(dataset_name)
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)

        try:
            file.save(file_path)
            return jsonify({
                'status': 'success',
                'message': f'Файл успешно загружен в {file_path}',
                'file_path': file_path
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка при сохранении файла: {str(e)}'
            }), 500

    @app.route('/results/<dataset_name>', methods=['GET'])
    def get_dataset_results(dataset_name):
        """
        Универсальный маршрут для получения результатов по имени датасета.
        """
        if dataset_name not in ALLOWED_DATASETS:
            return jsonify({
                'status': 'error',
                'message': f'Неверное имя датасета. Допустимые значения: {", ".join(ALLOWED_DATASETS)}'
            }), 400

        results_file = os.path.join(RESULTS_DIR, f'{dataset_name}_results.json')
        
        try:
            if not os.path.exists(results_file):
                return jsonify({
                    'status': 'error',
                    'message': f'Файл с результатами для датасета {dataset_name} не найден'
                }), 404

            with open(results_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return jsonify({
                'status': 'success',
                'dataset': dataset_name,
                'data': data
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка при чтении результатов: {str(e)}'
            }), 500

    @app.route('/inference/<dataset_name>', methods=['GET'])
    def run_model_inference(dataset_name):
        """
        Маршрут для запуска инференса для указанного датасета.
        """
        if dataset_name not in ALLOWED_DATASETS:
            return jsonify({
                'status': 'error',
                'message': f'Неверное имя датасета. Допустимые значения: {", ".join(ALLOWED_DATASETS)}'
            }), 400
        
        try:
            results = run_inference(dataset_name)
            return jsonify({
                'status': 'success',
                'message': f'Инференс для датасета {dataset_name} выполнен успешно',
                'results': results
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка при выполнении инференса: {str(e)}'
            }), 500

    @app.errorhandler(404)
    def not_found(error):
        """
        Обработчик ошибок для несуществующих маршрутов.
        """
        return jsonify({
            'status': 'error',
            'message': 'Маршрут не найден'
        }), 404

    @app.route('/list/<dataset_name>', methods=['GET'])
    def list_user_files(dataset_name):
        """
        Маршрут для получения списка файлов из папки !user для указанного датасета.
        """
        if dataset_name not in ALLOWED_DATASETS:
            return jsonify({
                "status": "error",
                "message": f"Неверное имя датасета. Допустимые значения: {', '.join(ALLOWED_DATASETS)}"
            }), 400

        upload_folder = get_upload_folder(dataset_name)

        # Проверяем существование папки
        if not os.path.exists(upload_folder):
            return jsonify({
                "status": "error",
                "message": f"Папка для загрузки файлов ({upload_folder}) не найдена."
            }), 404
        
        # Получаем список файлов
        try:
            files = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
            
            # Если файлов нет
            if not files:
                return jsonify({
                    "status": "success",
                    "message": "Файлы отсутствуют в папке.",
                    "files": []
                })
            
            # Возвращаем список файлов
            return jsonify({
                "status": "success",
                "message": f"Список файлов из папки !user для датасета {dataset_name}.",
                "files": files
            })
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Ошибка при получении списка файлов: {str(e)}"
            }), 500

    @app.route('/', methods=['GET'])
    def index():        
        """
        Корневой маршрут с информацией о доступных эндпоинтах.
        """
        return jsonify({
            'status': 'success',
            'message': f'API для работы с моделью классификации',
            "available_endpoints": {
                "POST /upload/<dataset_name>": "Загрузка файла",
                "GET /results/<dataset_name>": "Получение результатов",
                "GET /inference/<dataset_name>": "Запуск инференса"
            }
        })