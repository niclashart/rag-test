"""Structured logging configuration."""
import os
import sys
import structlog
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
    level=os.getenv("LOG_LEVEL", "INFO"),
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    serialize=os.getenv("LOG_FORMAT", "text") == "json"
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT", "text") == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger instance
def get_logger(name: str = __name__):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


