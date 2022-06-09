import logging.config

from flask import Flask, render_template, request, redirect, session, url_for
from flask_session import Session
from src.modeling import clean, predict
from src.manage_rds_db import PostManager

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile("config/flaskconfig.py")

# Set up flask session
SESSION_TYPE = app.config["SESSION_TYPE"]
SECRET_KEY = app.config["SECRET_KEY"]
Session(app)

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug(
    "Web app should be viewable at %s:%s if docker run command maps local "
    "port to the same port as configured for the Docker container "
    "in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the "
    "port defined on the left side of the port mapping "
    "(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, "
    "go to 127.0.0.1 instead of 0.0.0.0.", app.config["HOST"], app.config["PORT"])

# Initialize the database session
post_manager = PostManager(app)

# Retrieve the path to the folder containing 4 trained model object
MODEL_FOLDER = app.config["MODEL_FOLDER"]

# Retrieve the path to the vectorizer object
VECTORIZER_PATH = app.config["VECTORIZER_PATH"]

# Retrieve the path to the nltk packages
NLTK_DATA_PATH = app.config["NLTK_DATA_PATH"]


@app.route("/")
def home():
    """Main view that displays project headline and a button to input_text page

    Returns:
        Rendered index.html template
    """
    return render_template("index.html")


@app.route("/input_text", methods=["Get", "POST"])
def input_text():
    """View with a text box for entering texts.

    Returns:
        redirect to result page
    """
    if request.method == "POST":
        # Get the text from the text box
        text = request.form.get("text")
        session["text"] = text

        return redirect(url_for("show_result"))

    return render_template("search.html")


@app.route("/result", methods=["GET", "POST"])
def show_result():
    """View that displays the result of the text prediction.

    Returns:
        Rendered html template
    """
    # Get the text from the /input_text route
    try:
        text = session.get("text", None)
    except TypeError:
        logger.error("Failed to detect text entered.")
        return render_template("error.html")

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
    try:
        result_i = pred_result["I"].values[0]
    except UnboundLocalError as e:
        logger.error(
            "Failed to detect prediction result. Maybe vectorizer is not successfully generated. %s", e)
        return render_template("error.html")
    result_s = pred_result["S"].values[0]
    result_f = pred_result["F"].values[0]
    result_j = pred_result["J"].values[0]

    # Combine results to get a single MBTI type
    i_or_e = "I" if result_i > 0.5 else "E"
    s_or_n = "S" if result_s > 0.5 else "N"
    f_or_t = "F" if result_f > 0.5 else "T"
    j_or_p = "J" if result_j > 0.5 else "P"
    mbti_pred = i_or_e + s_or_n + f_or_t + j_or_p

    # Save raw user input, cleaned text, and predicted type to RDS database
    logger.info("Saving user input to RDS database")
    post_manager.ingest_app_user_input(raw_text=text, cleaned_text=cleaned_text,
                                       pred_type=mbti_pred, truncate=0)

    return render_template("result.html",
                           mbti_pred=mbti_pred,
                           result_I=result_i,
                           result_S=result_s,
                           result_F=result_f,
                           result_J=result_j)


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
