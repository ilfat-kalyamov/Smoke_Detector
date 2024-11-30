from modules.tg.config import dp, bot
from aiogram.methods import DeleteWebhook
from modules.tg.handlers import *

async def start_bot() -> None:
    await bot(DeleteWebhook(drop_pending_updates=True))
    await dp.start_polling(bot)