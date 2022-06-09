import os
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "MBTI_Predictor"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed

SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/posts.db'

# path to read locally saved model objects (folder) and vectorizer objects for predict.py
VECTORIZER_PATH = "output/vectorizer/tfidf_vectorizer.pkl"
MODEL_FOLDER = "output/model"

# path to nltk packages
NLTK_DATA_PATH = {"check_dl_nltk_data": {"download_dir": "data/nltk"}}

# flask-session config
SESSION_TYPE = "filesystem"
SECRET_KEY = "super secret key"
