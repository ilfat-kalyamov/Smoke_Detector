from modules.tg.config import dp, bot
from modules.tg.handlers import *

async def start_bot() -> None:
    await dp.start_polling(bot)