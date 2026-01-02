import logging
import sys
from pathlib import Path
import torch.distributed as dist

def setup_logger(log_path="logs/train.log", logger_name="InferenceLogger"):
    """
    Set up a logger that logs messages to both a file and console.

    Args:
        log_path (str or Path): Path to save the log file.
        logger_name (str): Name of the logger instance to ensure uniqueness.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_path = Path(log_path)  # Ensure it's a Path object
    log_dir = log_path.parent  # Extract directory path
    log_dir.mkdir(parents=True, exist_ok=True)  # Create logs directory if needed

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        # Console handler (prints to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # File handler (saves logs to a file)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)

        # Formatter (format log messages)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers only once
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Suppress duplicate logs in multi-GPU mode (Only rank 0 logs)
    if dist.is_initialized() and dist.get_rank() != 0:
        logger.setLevel(logging.WARNING)

    return logger
