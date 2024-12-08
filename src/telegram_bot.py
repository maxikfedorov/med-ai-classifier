import os
from dotenv import load_dotenv
import telebot

# Загружаем токен из .env
load_dotenv()
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))

MY_CHAT_ID = os.getenv('USER_ID')  

def send_notification(message_text: str) -> None:
    """Простая функция для отправки сообщения"""
    try:
        bot.send_message(MY_CHAT_ID, message_text)
        print(f"Сообщение отправлено: {message_text}")
    except Exception as e:
        print(f"Ошибка отправки сообщения: {e}")

def run_bot():
    """Запуск бота"""
    print("Бот запущен")
    bot.infinity_polling(none_stop=True)