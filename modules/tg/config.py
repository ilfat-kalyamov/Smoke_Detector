from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
import logging

load_dotenv(override=True)
BOT_TOKEN = getenv("BOT_TOKEN")
ADMIN_ID = getenv("ADMIN_ID")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("bot.log"),  # Запись логов в файл
            logging.StreamHandler()  # Вывод логов на консоль
        ]
    )

dp = Dispatcher()
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))