from modules.tg.admin_handler import admin_router
from modules.tg.config import ADMIN_ID, bot, dp
from modules.tg.user_handler import user_router
from modules.tg.config import on_error
from aiogram.methods import DeleteWebhook

async def start_bot() -> None:
    await bot(DeleteWebhook(drop_pending_updates=True))
    dp.include_routers(admin_router, user_router)
    dp.errors.register(on_error)
    await bot.send_message(chat_id=ADMIN_ID, text="Бот запущен!")
    await dp.start_polling(bot)