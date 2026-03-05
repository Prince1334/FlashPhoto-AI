import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Sets up Python logging for the Flask application."""
    # Ensure logs directory exists
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'flask_app.log')
    
    # 10 MB per file, keep 5 backups
    file_handler = RotatingFileHandler(log_file, maxBytes=10240000, backupCount=5)
    file_handler.setLevel(logging.INFO)
    
    # Format the log output
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to the Flask application logger
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("Application startup initialized.")
