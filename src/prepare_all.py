import os
from config import BASE_MODELS_DIR, CLASSIFICATION_MODEL_DIR,DATA_PATH

def main():
    os.makedirs(BASE_MODELS_DIR , exist_ok=True)
    os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

if __name__ == '__main__':
    main()

