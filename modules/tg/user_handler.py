import os
import logging

from aiogram import F, Router, html
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from modules.ai.config import load_model
from modules.ai.detect import predict_image, predict_transforms
from modules.tg.config import bot

device, model = load_model()
user_router = Router()

@user_router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!\nЯ нейросеть для распознавания факта курения.\n\nОтправь мне фотографию и я вынесу свой вердикт.\n\n/credits - информация")
    logging.info(f"User {message.from_user.id} used the comand /start")

@user_router.message(Command('credits'))
async def help_handler(message: Message) -> None:
    await message.answer(f'Проект разработан для дисциплины "{html.italic("Распознавание образов и машинное обучение")}".\n\nРазработчики: @for_what_or @h1dio')
    logging.info(f"User {message.from_user.id} used the comand /credits")

@user_router.message(F.photo)
async def image_handler(message: Message) -> None:
    #await message.reply("Фото получено...")
    file_id = message.photo[-1].file_id
    file_name = str(message.chat.id) + "_" + str(message.message_id) + ".png"
    await bot.download(file=file_id, destination=f"work/{file_name}")

    label = predict_image(f"work/{file_name}", model, predict_transforms, device)
    os.remove(f"work/{file_name}")
    await message.reply(f"Вердикт: {label}")
    logging.info(f"User {message.from_user.id} used the predict method with image {file_id}")

@user_router.message()
async def other_handler(message: Message) -> None:
    await message.answer("Я могу обрабатывать только фотографии, либо сообщения с фотографиями.")
    logging.info(f"User {message.from_user.id} used the other method")