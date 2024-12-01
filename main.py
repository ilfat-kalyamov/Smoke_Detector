import asyncio
import logging
import sys
from modules.ai.config import load_model

from modules.tg.bot import start_bot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(start_bot())