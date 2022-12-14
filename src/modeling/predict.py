import logging
import os
import pickle
import re
from typing import Tuple

import pandas as pd
from pandas.errors import ParserError
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


class IncorrectNumberOfFilesError(Exception):
    """Definie a custom exception class for incorrect number of files in
    a folder"""
    pass  # pylint: disable=unnecessary-pass


class IncorrectFilenameError(Exception):
    """Define a custom exception class for incorrect filename"""
    pass  # pylint: disable=unnecessary-pass


def read_new_text(text_path: str) -> str:
    """Read the data to be predicted from the given .csv file.

    Args:
        text_path (`str`): Path to the csv file

    Returns:
        pandas DataFrame of the data to be predicted.

    Raises:
        FileNotFoundError: If the text file does not exist.
        ParserError: If the text file fails to parse.
    """
    # Read the new text from the given file
    try:
        logger.info("Reading new text from file: %s", text_path)
        text = pd.read_csv(text_path, encoding="ISO-8859-1")
    except FileNotFoundError as e:
        logger.error("File not found: %s", text_path)
        raise e
    except ParserError as e:
        logger.error("Error parsing text file: %s", text_path)
        raise e
    else:
        logger.info("New text read.")

    return text


def validate_model_folder(model_file_folder: str) -> None:
    """Validate that the given model folder exists, contains exactly 4 files,
    each of which is a pickle file, and is named in the format of `logit_<col>=1.pkl`

    Args:
        model_folder (`str`): Path to the model folder.

    Returns:
        None

    Raises:
        FileNotFoundError: If the model folder does not exist.
        IncorrectNumberOfFilesError: If the model folder doesn't contain exactly 4 files.
        IncorrectFilenameError: If the model folder contains files with incorrect names.
        TypeError: If the model folder contains files other than pickle files.
    """
    logger.info("Validating model folder: %s", model_file_folder)
    # Validate that the model folder exists
    if not os.path.exists(model_file_folder):
        logger.error("Model folder does not exist: %s",
                     model_file_folder)
        raise FileNotFoundError

    # Validate that the model folder contains exactly 4 files
    model_files = os.listdir(model_file_folder)
    if len(model_files) != 4:
        logger.error("Model folder should contain exactly 4 files, "
                     "each for 1 MBTI dimension: %s",
                     model_file_folder)
        raise IncorrectNumberOfFilesError(
            f"Model folder should contain exactly 4 files. {len(model_files)} detected")

    # Validate that each file is a pickle file and is named in the format of `logit_<col>=1.pkl`
    for model_file in model_files:
        logger.debug("Validating model file: %s", model_file)
        if not model_file.endswith(".pkl"):
            logger.error("Model file should be a pickle file: %s",
                         model_file)
            raise TypeError("Model file should be a pickle file. "
                            f"{os.path.splitext(model_file)[-1]} detected")
        if not re.match(r"logit_[ISFJ]=1.pkl", model_file):
            logger.error("Model file should be named in the format of `logit_<col>=1.pkl`, "
                         "where col is one of [I, S, F, J]: %s", model_file)
            raise IncorrectFilenameError(f"Model filename {model_file} is invalid. Supported format is "
                                         "`logit_<col>=1.pkl`, where col is one of [I, S, F, J].")


def read_model_from_path(model_file_path: str) -> LogisticRegression:
    """Read the model object from the pickle file.

    Args:
        model_path (`str`): Path to the pickle file containing the model object.

    Returns:
        The LogisticRegression model object.

    Raises:
        FileNotFoundError: If the model pickle file does not exist.
        EOFError, UnpicklingError: If the model pickle file fails to unpickle.
    """

    # Load model objects from pickle file
    try:
        logger.info("Reading model from pickle file: %s", model_file_path)
        model_file = open(model_file_path, "rb")
    except FileNotFoundError as e:
        logger.error("File not found: %s", model_file_path)
        raise e
    else:
        try:
            model = pickle.load(model_file)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error("Error reading model from pickle file: %s",
                         model_file_path)
            raise e
        else:
            logger.info("Model object loaded.")
        finally:
            model_file.close()
    return model


def read_vectorizer_from_path(vectorizer_path: str) -> TfidfVectorizer:
    """Read the vectorizer from given path.

    Args:
        vectorizer_path (`str`): Path to the vectorizer file saved in previous train step.

    Returns:
        The TfidfVectorizer object.

    Raises:
        FileNotFoundError: If the vectorizer file does not exist.
        EOFError, UnpicklingError: If the vectorizer file fails to unpickle.
    """

    # Load vectorizer saved in previous step
    try:
        logger.info("Reading vectorizer from pickle file: %s", vectorizer_path)
        vectorizer_file = open(vectorizer_path, "rb")
    except FileNotFoundError as e:
        logger.error("File not found: %s", vectorizer_path)
        raise e
    else:
        try:
            vectorizer = pickle.load(vectorizer_file)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error("Error reading model from pickle file: %s",
                         vectorizer_file)
            raise e
        else:
            logger.info("Vectorizer object loaded.")
        finally:
            vectorizer_file.close()
    return vectorizer


def predict_logit(model: LogisticRegression, new_text: str) -> Tuple[list, list]:
    """Predict the class probabilities and class predictions on the new text.

    Args:
        model (`sklearn.linear_model.LogisticRegression`): The LogisticRegression model object.
        new_text (`str`): The new text to predict.

    Returns:
        A tuple of list, containing the predicted probabilities and predicted classes

    Raises:
        ValueError: Errors when predicting the model on the new_text,
            such as sparse matrix provided.
    """

    # Predict the probabilities of the new text belonging to each class
    try:
        logger.info("Predicting class probabilities on the new text: %s...")
        y_pred_prob = model.predict_proba(new_text)[:, 1]
    except ValueError as e:
        logger.error("Error predicting class probabilities on: %s",
                     new_text)
        raise e
    except Exception as e:
        logger.error("Unknown error predicting class probabilities  on: %s",
                     new_text)
        raise e

    # Predict the class of the new text
    try:
        logger.info("Predicting class on the new text: %s...")
        y_pred_bin = model.predict(new_text)
    except ValueError as e:
        logger.error("Error predicting class on: %s",
                     new_text)
        raise e
    except Exception as e:
        logger.error("Unknown error predicting class on: %s",
                     new_text)
        raise e

    return y_pred_prob, y_pred_bin


def save_ypred_to_files(y_pred_df: pd.DataFrame,
                        y_pred_filename: str,
                        y_pred_output_dir: str) -> None:
    """Save the dataframe that contains prediction probabilities and classes to
    the given path.

    Args:
        y_pred_df (`pd.DataFrame`): The dataframe that contains prediction probabilities and classes.
        y_pred_filename (`str`): The filename of file to save the dataframe.
        y_pred_output_dir (`str`): The output directory of the file.

    Returns:
        None

    Raises:
        PermissionError: If the output directory is not writable.
    """
    logger.info("Saving predictions to files.")

    # Validate ypred_output_dir exists. If not, create a new directory at
    # ypred_output_dir.
    if not os.path.exists(y_pred_output_dir):
        logger.warning("Output directory does not exist: %s. \
        Creating new directory.", y_pred_output_dir)
        os.makedirs(y_pred_output_dir)
        logger.info("Created directory: %s",
                    y_pred_output_dir)

    # Save predictions to files as csv files
    output_path = os.path.join(y_pred_output_dir, y_pred_filename) + ".csv"
    try:
        y_pred_df.to_csv(output_path, index=False)
    except PermissionError as e:
        logger.error("Permission denied to write to file: %s",
                     output_path)
        raise e
    except Exception as e:
        logger.error("Unknown error writing to file: %s", output_path)
        raise e

    logger.info("Predictions saved to directory %s",
                output_path)


def predict_wrapper(model_folder_path: str, new_text_path: str, vectorizer_path: str,
                    y_pred_output_dir: str, is_string: bool, save_output: bool,
                    **kwargs_predict) -> pd.DataFrame:
    """Wrapper function to predict the class probabilities on the new text.

    Saved model object, vectorizer object, and new texts are loaded from the given paths.
    New texts could either be the test set saved from previous step, or a string from
    user input. The new texts are then predicted on the model, and the prediction
    probabilities and classes are saved to the given output directory.

    Args:
        model_folder_path (`str`): Path to the folder that contains the model pickle files.
        new_text_path (`str`): Either path to the file that contains the new texts or a text string
        vectorizer_path (`str`): Path to the vectorizer file saved in previous train step.
        y_pred_output_dir (`str`): Path to save the prediction probabilities and classes.
        is_string (`bool`): Whether the `new_text_path` is a string or a file path.
        save_output (`bool`): Whether to save the prediction probabilities and classes to files.
        kwargs_predict (`dict`): Dictionary `predict_wrapper` defined in config.yaml
            - posts_column (`str`): Column name of posts in new texts dataframe
            - y_pred_filename_prefix (`str`): Prefix of the filename of the predictions

    Returns:
        A pandas DataFrame that contains the prediction probabilities for each dimension,
        "I", "S", "F", "J".

    Raises:
        KeyError: If the `posts_column` is not in the new texts dataframe.
    """

    logger.info("Starting predict.py pipeline")

    # Create a dataframe to store all 4 predictions separately for later evaluation
    class_result = pd.DataFrame(columns=["probability", "class"])

    # Create a dataframe for all predictions for new text prediction
    result_prob = pd.DataFrame(columns=["I", "S", "F", "J"])

    # Validate model files in the given folder
    validate_model_folder(model_folder_path)

    # Read the new text to be predicted
    if not is_string:
        new_text = read_new_text(new_text_path)
    else:
        new_text = new_text_path

    # Vectorize the new text
    vectorizer = read_vectorizer_from_path(vectorizer_path)
    if not is_string:
        try:
            posts = new_text[kwargs_predict["posts_colname"]].tolist()
        except KeyError as e:
            logger.error("Errors reading posts column. Please check name of the target "
                         "column in dataset matches with what's been defined in config: %s", e)
            raise e
        else:
            new_text_vectorized = vectorizer.transform(posts).toarray()
    else:
        posts = new_text
        new_text_vectorized = vectorizer.transform([posts]).toarray()

    # Iterate over all the model files in the model_file_folder
    for model_filename in os.listdir(model_folder_path):
        # Get the model file path
        model_file_path = os.path.join(model_folder_path, model_filename)

        # Read the model object from the pickle file
        model = read_model_from_path(model_file_path)

        # Predict the class probabilities on the new text
        y_pred_prob, y_pred_bin = predict_logit(model, new_text_vectorized)

        # Save to result dataframe
        # model_filename[6] is one of [I, S, F, J]
        # y_pred_prob[1] is the probability of belonging to class 1
        class_result["probability"] = y_pred_prob
        class_result["class"] = y_pred_bin
        result_prob[model_filename[6]] = y_pred_prob

        if save_output:
            # Save the predictions to files if specified
            y_pred_filename_prefix = kwargs_predict["y_pred_filename_prefix"]
            save_ypred_to_files(class_result,
                                f"{y_pred_filename_prefix}_{model_filename[6]}=1",
                                y_pred_output_dir)

    return result_prob
