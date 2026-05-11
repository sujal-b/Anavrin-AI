import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    
    # Try to get the specific key based on provider, fallback to generic LLM_API_KEY
    if LLM_PROVIDER.upper() == "OPENROUTER":
        LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    elif LLM_PROVIDER.upper() == "NVIDIA":
        LLM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    else:
        LLM_API_KEY = os.getenv("LLM_API_KEY", "")

    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

    # Model loading priority (best → worst)
    MODEL_PRIORITY = ["naive_bayes", "knn", "random_forest"]


config = Config()
