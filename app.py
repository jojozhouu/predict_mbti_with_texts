import logging.config
import sqlite3
import traceback

import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, url_for

# For setting up the Flask-SQLAlchemy database session
from src.manage_rds_db import PostManager

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

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
track_manager = PostManager(app)


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
    # Get the text from the text box
    text = request.form.get('text')
    # TODO call preprocess and predict functions
    return render_template('search.html')
    # try:
    #     track_manager.add_track(artist=request.form['artist'],
    #                             album=request.form['album'],
    #                             title=request.form['title'])
    #     logger.info("New song added: %s by %s", request.form['title'],
    #                 request.form['artist'])
    #     return redirect(url_for('index'))
    # except sqlite3.OperationalError as e:
    #     logger.error(
    #         "Error page returned. Not able to add song to local sqlite "
    #         "database: %s. Error: %s ",
    #         app.config['SQLALCHEMY_DATABASE_URI'], e)
    #     return render_template('error.html')
    # except sqlalchemy.exc.OperationalError as e:
    #     logger.error(
    #         "Error page returned. Not able to add song to MySQL database: %s. "
    #         "Error: %s ",
    #         app.config['SQLALCHEMY_DATABASE_URI'], e)
    #     return render_template('error.html')


@app.route("/result", methods=['GET', 'POST'])
def show_result():
    """View that displays the result of the text prediction.

    Returns:
        Rendered html template
    """
    return render_template('result.html')


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
