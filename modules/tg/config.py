import os
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
import logging
from logging.handlers import RotatingFileHandler

load_dotenv(override=True)
BOT_TOKEN = getenv("BOT_TOKEN")
ADMIN_ID = getenv("ADMIN_ID")

os.makedirs("logs", exist_ok=True)
log_file = "logs/bot.log"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        ]
    )

dp = Dispatcher()
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

async def on_error(update, exception):
    logging.error(f"Update {update} caused error {exception}")
    bot.send_message(chat_id=ADMIN_ID, text=f"Update {update} caused error {exception}")