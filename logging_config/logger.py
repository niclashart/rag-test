"""Structured logging configuration."""
import os
import sys
import structlog
import logging
from loguru import logger
from pathlib import Path

# Create logs directory
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default loguru handler
logger.remove()

# Add console handler with structured output
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
    colorize=True
)

# Add file handler with JSON format
logger.add(
    LOG_DIR / "app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level=os.getenv("LOG_LEVEL", "DEBUG"),  # Set to DEBUG to capture more info
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    serialize=True  # Force JSON for file
)

# Add interaction file handler
logger.add(
    LOG_DIR / "interactions.log",
    format="{message}",  # We will log raw JSON string
    filter=lambda record: record["extra"].get("type") == "interaction",
    level="INFO",
    rotation="10 MB",
    retention="90 days",
    compression="zip",
    serialize=False # We handle serialization manually or let the message be the JSON
)

# Configure structlog to wrap loguru
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT", "text") == "json" else structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Redirect standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Get logger instance
def get_logger(name: str = __name__):
    """Get a structured logger instance."""
    # Return a structlog logger that uses standard logging, which is intercepted by loguru
    return structlog.get_logger(name)


