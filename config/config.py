import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration"""
    PROJECT_NAME = "SmartRecommendation"
    PROJECT_VERSION = "1.0.0"
    
    # Data paths
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    TEST_DATA_PATH = "data/test"
    
    # Model paths
    MODEL_PATH = "models"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartrec.db")
    
    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Redis (for caching)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/app.log"

config = Config()
