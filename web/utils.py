import os
import re

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed_data')

def allowed_file(filename):
    """Проверка допустимого расширения файла."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def custom_secure_filename(filename):
    """
    Обрабатывает имя файла, оставляя кириллицу и заменяя недопустимые символы на '_'.
    """
    filename = re.sub(r'[^\w.\-а-яА-ЯёЁ]', '_', filename)  # Разрешаем буквы, цифры, точки, дефисы и кириллицу
    return filename

def get_upload_folder(dataset_name):
    """Получение пути для загрузки файлов."""
    return os.path.join(DATA_DIR, dataset_name, '!user')