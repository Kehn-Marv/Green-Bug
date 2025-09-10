import logging
import sys

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup structured logging for the application"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def log_analysis_request(logger: logging.Logger, filename: str, file_size: int):
    """Log incoming analysis request"""
    logger.info(f"Analysis request: {filename} ({file_size} bytes)")

def log_analysis_result(logger: logging.Logger, result: dict):
    """Log analysis results"""
    face_found = result.get("face", {}).get("found", False)
    deep_score = result.get("scores", {}).get("deep_model_score")
    logger.info(
        f"Analysis complete: face={face_found}, "
        f"deep_score={deep_score if deep_score is not None else 'N/A'}"
    )

def log_error(logger: logging.Logger, error: Exception, context: str):
    """Log errors with context"""
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
