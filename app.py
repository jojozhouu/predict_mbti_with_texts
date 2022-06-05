from datetime import datetime
import logging.config
import os
import sqlite3
import traceback

import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, session, url_for
from flask_session import Session
from config.flaskconfig import SECRET_KEY, SESSION_TYPE

from src.modeling import clean, predict
from src import manage_rds_db

# For setting up the Flask-SQLAlchemy database session
from src.manage_rds_db import PostManager

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Set up flask session
SESSION_TYPE = app.config["SESSION_TYPE"]
SECRET_KEY = app.config["SECRET_KEY"]
Session(app)

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug(
    'Web app should be viewable at %s:%s if docker run command maps local '
    'port to the same port as configured for the Docker container '
    'in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the '
    'port defined on the left side of the port mapping '
    '(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, '
    'go to 127.0.0.1 instead of 0.0.0.0.', app.config["HOST"], app.config["PORT"])

# Initialize the database session
post_manager = PostManager(app)

# Retrieve the path to the folder containing 4 trained model object
MODEL_FOLDER = app.config["MODEL_FOLDER"]

# Retrieve the path to the vectorizer object
VECTORIZER_PATH = app.config["VECTORIZER_PATH"]

# Retrieve the path to the nltk packages
NLTK_DATA_PATH = app.config["NLTK_DATA_PATH"]

# Specify paths to output clean data and stopwords
# CLEAN_DATA_PATH = app.config["CLEAN_DATA_PATH"]
# CLEAN_DATA_FILENAME = app.config["CLEAN_DATA_FILENAME"]
# STOPWORDS_PATH = app.config["STOPWORDS_PATH"]
# STOPWORDS_FILENAME = app.config["STOPWORDS_FILENAME"]

# # Retrieve S3 path to save clean data to, if SAVE_OUTPUT_TO_S3 is True
# SAVE_OUTPUT_TO_S3 = app.config["SAVE_OUTPUT_TO_S3"]
# S3_PATH = app.config["S3_PATH"]
# CLEAN_DATA_PATH = app.config["CLEAN_DATA_PATH"]

# S3 path to save clean data to


@app.route('/')
def home():
    """Main view that displays all 16 MBTI types and a button to input_text page

    Returns:
        Rendered html template
    """
    return render_template('index.html')

# @app.route('/')
# def index():
#     """Main view that displays all 16 MBTI types and a text box for entering texts.

#     Returns:
#         Rendered html template
#     """

#     try:
#         tracks = track_manager.session.query(Tracks).limit(
#             app.config["MAX_ROWS_SHOW"]).all()
#         logger.debug("Index page accessed")
#         return render_template('index.html', tracks=tracks)
#     except sqlite3.OperationalError as e:
#         logger.error(
#             "Error page returned. Not able to query local sqlite database: %s."
#             " Error: %s ",
#             app.config['SQLALCHEMY_DATABASE_URI'], e)
#         return render_template('error.html')
#     except sqlalchemy.exc.OperationalError as e:
#         logger.error(
#             "Error page returned. Not able to query MySQL database: %s. "
#             "Error: %s ",
#             app.config['SQLALCHEMY_DATABASE_URI'], e)
#         return render_template('error.html')
#     except:
#         traceback.print_exc()
#         logger.error("Not able to display tracks, error page returned")
#         return render_template('error.html')


@app.route('/input_text', methods=['Get', 'POST'])
def input_text():
    """View that process a text box for entering texts.

    Returns:
        redirect to result page
    """
    if request.method == 'POST':
        # Get the text from the text box
        text = request.form.get('text')
        session["text"] = text

        return redirect(url_for('show_result'))

    return render_template("search.html")


@app.route("/result", methods=['GET', 'POST'])
def show_result():
    """View that displays the result of the text prediction.

    Returns:
        Rendered html template
    """
    # Get the text from the /input_text route
    try:
        text = session.get('text', None)
    except TypeError:
        logger.error("Failed to detect text entered.")
        return render_template('error.html')

    # Process the text
    cleaned_text = clean.clean_wrapper(raw_data=text,
                                       clean_data_output_dir=None,
                                       is_new_data=True,
                                       save_output=False,
                                       **NLTK_DATA_PATH)

    # Predict with pre-trained model object
    pred_result = predict.predict_wrapper(model_folder_path=MODEL_FOLDER,
                                          new_text_path=cleaned_text,
                                          vectorizer_path=VECTORIZER_PATH,
                                          y_pred_output_dir=None,
                                          is_string=True,
                                          save_output=False)

    # Retrieve prediction results and send to template for display
    result_I = pred_result["I"].values[0]
    result_S = pred_result["S"].values[0]
    result_F = pred_result["F"].values[0]
    result_J = pred_result["J"].values[0]

    # Combine results to get a single MBTI type
    IorE = "I" if result_I > 0.5 else "E"
    SorN = "S" if result_S > 0.5 else "N"
    ForT = "F" if result_F > 0.5 else "T"
    JorP = "J" if result_J > 0.5 else "P"
    mbti_pred = IorE + SorN + ForT + JorP

    # Save raw user input, cleaned text, and predicted type to RDS database
    logger.info("Saving user input to RDS database")
    post_manager.ingest_app_user_input(raw_text=text, cleaned_text=cleaned_text,
                                       pred_type=mbti_pred, truncate=0)
    logger.info("Raw user input saved to RDS database")

    return render_template('result.html',
                           mbti_pred=mbti_pred,
                           result_I=result_I,
                           result_S=result_S,
                           result_F=result_F,
                           result_J=result_J)


@app.route('/learn_more')
def learn_more():
    """View that process a text box for entering texts.

    Returns:
        redirect to result page
    """
    # TODO call preprocess and predict functions
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
