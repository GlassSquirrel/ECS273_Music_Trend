"""
logger.py
=========
Shared logging utility. Import and call setup_logger() at the top of
each script to write all stdout-level logs to both the console and a
timestamped file under ml/logs/.

Usage:
    from logger import setup_logger
    logger = setup_logger(__name__)   # in data_merge.py, train_vae.py, etc.
    logger.info("Step 1: Loading data ...")
"""

import logging
import os
from datetime import datetime

LOGS_DIR = "logs"

def setup_logger(name: str) -> logging.Logger:
    """
    Create a logger that writes to both console and a timestamped log file.

    The log file is named <name>_<YYYYMMDD_HHMMSS>.log and saved to ml/logs/.
    The script name (e.g. 'data_merge') is used as both the logger name and
    the filename prefix, so each run produces its own file.

    Parameters
    ----------
    name : typically pass __name__, or the script stem e.g. 'train_vae'

    Returns
    -------
    logger : logging.Logger
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Use just the module stem as prefix (e.g. 'data_merge', not '__main__')
    stem = os.path.splitext(os.path.basename(name))[0] if os.sep in name else name
    if stem == "__main__":
        # Fallback: infer from call stack
        import inspect
        stem = os.path.splitext(
            os.path.basename(inspect.stack()[-1].filename)
        )[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOGS_DIR, f"{stem}_{timestamp}.log")

    logger = logging.getLogger(stem)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    # Full format with timestamp and level -- used for the log file
    file_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Plain format -- terminal output looks identical to print()
    console_fmt = logging.Formatter(fmt="%(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger