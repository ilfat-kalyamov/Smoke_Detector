import logging
from aiogram import html
from aiogram.filters import BaseFilter, Command
from aiogram.types import Message, FSInputFile
from modules.tg.config import ADMIN_ID, log_file

from aiogram import Router

admin_router = Router()

class IsAdmin(BaseFilter):
    def __init__(self) -> None:
        self.admin_id = int(ADMIN_ID)
    async def __call__(self, message: Message) -> bool:
        return message.from_user.id == int(ADMIN_ID)

@admin_router.message(IsAdmin(), Command('start'))
async def admin_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!\nЯ нейросеть для распознавания факта курения.\n\nОтправь мне фотографию и я вынесу свой вердикт.\n\n/credits - информация\n\nВам доступна админ панель:\n/admin - Команды администрирования")
    logging.info(f"Admin {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the ADMIN comand /start")

@admin_router.message(IsAdmin(), Command('admin'))
async def admin_handler(message: Message) -> None:
    await message.answer("Админ панель:\n\n/logs - отправить лог-файл")
    logging.info(f"Admin {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the ADMIN comand /admin")

@admin_router.message(IsAdmin(), Command('logs'))
async def log_sender(message: Message) -> None:
    try:
        logs = FSInputFile(log_file)
        await message.answer_document(logs)
    except Exception as e:
        await message.answer("Не удалось отправить лог-файл")
    logging.info(f"Admin {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the ADMIN comand /logs")