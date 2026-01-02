# fusion_toolbox/utils/logging_setup.py

import logging
from pathlib import Path

def get_logger(name=__name__, log_file="output/compounding.log"):
    # Ensure that the directory exists:
    log_path = Path(log_file)
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler()
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
