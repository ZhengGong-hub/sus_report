import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# save the logger to a file
from datetime import datetime

log_filename = datetime.now().strftime("logs/logs_%Y%m%d_%H%M.log")

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s",
    "%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)