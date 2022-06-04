from datetime import datetime
import os
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "penny-lane"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100

SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tracks.db'


# path to read locally saved model objects (folder) and vectorizer objects for predict.py
VECTORIZER_PATH = "output/vectorizer/tfidf_vectorizer.pkl"
MODEL_FOLDER = "output/model"

# path to nltk packages
NLTK_DATA_PATH = {"check_dl_nltk_data": {"download_dir": "data/nltk"}}

# flask-session config
SESSION_TYPE = "filesystem"
SECRET_KEY = "super secret key"

# # path to locally save clean data and stopwords to in clean.py
# CLEAN_DATA_PATH = "data/clean"
# CLEAN_DATA_FILENAME = f"/clean_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv"
# STOPWORDS_PATH = "data/stopwords"
# STOPWORDS_FILENAME = "stopwords.csv"


# # S3 path to save clean data to
# S3_PATH = "s3://2022-msia423-zhou-jojo"

# # Upload clean data and stopwords to S3 every time the app is run, if True
# SAVE_OUTPUT_TO_S3 = False
# CLEAN_DATA_PATH = S3_PATH + "/clean" + f"/clean_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.csv")
