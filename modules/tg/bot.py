from modules.tg.config import dp, bot
from aiogram.methods import DeleteWebhook
from modules.tg.handlers import *
from modules.tg.config import ADMIN_ID

async def start_bot() -> None:
    await bot(DeleteWebhook(drop_pending_updates=True))
    await bot.send_message(chat_id=ADMIN_ID, text="Бот запущен!")
    await dp.start_polling(bot)