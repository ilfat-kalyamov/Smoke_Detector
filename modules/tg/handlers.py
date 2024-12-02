from aiogram import html, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

import os

from modules.ai.config import load_model
from modules.ai.detect import predict_image, predict_transforms

from modules.tg.config import bot, dp

device, model = load_model()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!\nЯ нейросеть для распознавания факта курения.\n\nОтправь мне фотографию и я вынесу свой вердикт.\n\n/credits - информация")

@dp.message(Command('credits'))
async def help_handler(message: Message) -> None:
    await message.answer(f'Проект разработан для дисциплины "{html.italic("Распознавание образов и машинное обучение")}".\n\nРазработчики: @for_what_or @h1dio')

@dp.message(F.photo)
async def image_handler(message: Message) -> None:
    await message.reply("Фото получено...")
    file_id = message.photo[-1].file_id
    await bot.download(file=file_id, destination="work/file.png")

    label = predict_image("work/file.png", model, predict_transforms, device)
    os.remove("work/file.png")
    await message.reply(f"Предсказанный класс: {label}")

@dp.message()
async def echo_handler(message: Message) -> None:
    await message.answer("Я могу обрабатывать только фотографии, либо сообщения с фотографиями.")