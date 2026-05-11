import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

    # Model loading priority (best → worst)
    MODEL_PRIORITY = ["naive_bayes", "knn", "random_forest"]


config = Config()
