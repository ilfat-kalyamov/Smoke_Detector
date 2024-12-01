import asyncio
import logging
import sys
from modules.tg.config import setup_logging

from modules.tg.bot import start_bot

if __name__ == "__main__":
    setup_logging()
    asyncio.run(start_bot())