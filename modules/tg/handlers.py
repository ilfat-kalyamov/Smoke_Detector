from aiogram import html, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

from modules.tg.config import dp

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!\nЯ нейросеть для распознавания факта курения.\n\nОтправь мне фотографию и я вынесу свой вердикт.\n\n/help - помощь")

@dp.message(Command('help'))
async def help_handler(message: Message) -> None:
    await message.answer("Разработчик: @for_what_or")

@dp.message(F.photo)
async def image_handler(message: Message) -> None:
    await message.answer("Фото получено...")
    answer = message.photo[0].file_id
    await message.answer(answer)

@dp.message()
async def echo_handler(message: Message) -> None:
    await message.answer("Я могу обрабатывать только фотографии, либо сообщения с фотографиями.")