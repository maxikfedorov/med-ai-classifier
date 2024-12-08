from flask import Flask
from flask_cors import CORS
import os
from routes import register_routes

# Константы
FLASK_DEBUG = True
FLASK_PORT = 5000

# Инициализация Flask
app = Flask(__name__)
CORS(app)

# Максимальный размер файла (16 МБ)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Регистрация маршрутов
register_routes(app)

if __name__ == '__main__':
    # Создаем директорию для результатов, если она не существует
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Сервер запущен на порту {FLASK_PORT}")
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)