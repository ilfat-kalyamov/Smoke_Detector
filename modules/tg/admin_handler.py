from aiogram.filters import BaseFilter, Command
from aiogram.types import Message
from aiogram import F
from modules.tg.config import ADMIN_ID

from aiogram import Router

admin_router = Router()

class IsAdmin(BaseFilter):
    def __init__(self) -> None:
        self.admin_id = int(ADMIN_ID)

    async def __call__(self, message: Message) -> bool:
        return message.from_user.id == int(ADMIN_ID)


@admin_router.message(IsAdmin(), Command('admin'))
async def admin_handler(message: Message) -> None:
    await message.answer("Админ обработчик сработал")