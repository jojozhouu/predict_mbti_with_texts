import logging
import os
import pickle
from typing import Tuple

import pandas as pd
from pandas.errors import ParserError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def read_test_from_path(y_test_path: str) -> pd.Series:
    """
    Read the actual target classes of the test data from local file.

    Args:
        y_test_path (`str`): Path to the y_test file saved in previous train_test_split.

    Returns:
        A pandas Series containing y_test.

    Raises:
        FileNotFoundError: If the file is not found.
        ParserError: If the file fails to parse.
        IOError, OSError: If the file is not readable.
    """
    logger.info("Retrieving y_test from previous train-test-split from file.")
    # Load y_test saved in previous step
    try:
        y_test = pd.read_csv(y_test_path)
        logger.info("y_test loaded from %s.", y_test_path)
    except FileNotFoundError as fe:
        logger.error("File not found: %s", y_test_path)
        raise fe
    except ParserError as pe:
        logger.error("Error parsing data from %s", y_test_path)
        raise pe
    except Exception as e:
        logger.error("Unknown error reading data from %s", y_test_path)
        raise e

    return y_test


def read_pred_from_path(y_pred_path: str) -> list:
    """
    Read the predicted classes of the test data from local file.

    Args:
        y_pred_path (`str`): Path to the y_pred file saved in previous scoring.

    Returns:
        A pandas Series containing y_pred.

    Raises:
        FileNotFoundError: If the file is not found.
        ParserError: If the file fails to parse.
        IOError, OSError: If the file is not readable.
    """
    logger.info("Retrieving y_pred from previous prediction.")
    # Load y_pred saved in previous step
    try:
        y_pred = pd.read_csv(y_pred_path)
        logger.info("y_pred loaded from %s.", y_pred_path)
    except FileNotFoundError as fe:
        logger.error("File not found: %s", y_pred_path)
        raise fe
    except ParserError as pe:
        logger.error("Error parsing data from %s", y_pred_path)
        raise pe
    except Exception as e:
        logger.error("Unknown error reading data from %s", y_pred_path)
        raise e

    y_pred_bin = y_pred["class"].tolist()

    return y_pred_bin


# def read_vectorizer_from_path(vectorizer_path: str) -> TfidfVectorizer:
#     """
#     Read the vectorizer from local pickle file.

#     Args:
#         vectorizer_path (`str`): Path to the vectorizer file saved in previous train step.
#     """
#     # Load vectorizer saved in previous step
#     try:
#         logger.info("Reading vectorizer from pickle file: %s", vectorizer_path)
#         vectorizer_file = open(vectorizer_path, "rb")
#     except FileNotFoundError as e:
#         logger.error("File not found: %s", vectorizer_path)
#         raise e
#     else:
#         try:
#             vectorizer = pickle.load(vectorizer_file)
#         except (EOFError, pickle.UnpicklingError) as e:
#             logger.error("Error reading model from pickle file: %s",
#                          vectorizer_file)
#             raise e
#         else:
#             logger.info("Vectorizer object loaded.")
#         finally:
#             vectorizer_file.close()
#     return vectorizer

def verify_test_and_pred(y_test: pd.Series, y_pred: list) -> bool:
    """
    Verify if the test and prediction data are the same length.

    Args:
        y_test (`pd.Series`): Actual target classes.
        ypred_bin (`list`): Predicted target classes.

    Returns:
        True if the test and prediction data are the same length.

    Raises:
        None
    """
    logger.info("Verifying test and prediction data.")
    # Verify if the test and prediction data are the same length
    if len(y_test.index) != len(y_pred):
        logger.error("Test and prediction data are not of the same length.")
        raise ValueError(
            "Test and prediction data are not of the same length.")
    # TODO , add more checks


def confusion_mat(y_test: pd.Series, ypred_bin: list) -> pd.DataFrame:
    """
    Calculate the confusion matrix between the actual and predicted test target data.

    Args:
        y_test (`pd.Series`): Actual target classes.
        ypred_bin (`list`): Predicted target classes.

    Returns:
        A pandas DataFrame containing the confusion matrix.

    Raises:
        None
    """
    logger.info("Calculating confusion matrix.")
    # Calculate the confusion matrix
    try:
        mat = confusion_matrix(y_test, ypred_bin)

        # Convert to pandas dataframe for later print-out
        confusion_df = pd.DataFrame(mat,
                                    index=["Actual negative",
                                           "Actual positive"],
                                    columns=["Predicted negative", "Predicted positive"])
    except ValueError as e:
        logger.warning(
            "Error calculating confusion matrix. "
            "Skipped in metrics report. %s", e)
        return "N/A"

    return confusion_df


def accuracy(y_test: pd.Series, ypred_bin: list) -> float:
    """
    Calculate accuracy score.

    Args:
        y_test (`pd.Series`): Actual target classes.
        ypred_bin (`list`): Predicted target classes.

    Returns:
        Float type accuracy score.

    Raises:
        None
    """
    logger.info("Calculating accuracy score.")
    # Calculate the accuracy score
    try:
        acc = accuracy_score(y_test, ypred_bin)
    except ValueError as e:
        logger.warning(
            "Error calculating accuracy score. Please check the formats of "
            "y_test and y_pred. Skipped in metrics report. %s", e)
        return "N/A"

    return acc


def class_report(y_test: pd.Series, ypred_bin: list) -> pd.DataFrame:
    """
    Generate the classification report.

    Args:
        y_test (`pd.Series`): Actual target classes.
        ypred_bin (`list`): Predicted target classes.

    Returns:
        A pandas DataFrame containing the classification report.

    Raises:
        None
    """
    logger.info("Generating classification report.")
    # Generate the classification report
    try:
        report = classification_report(y_test, ypred_bin, output_dict=True)
    except ValueError as e:
        logger.warning(
            "Error calculating classification report. "
            "Skipped in metrics report. %s", e)
        return "N/A"

    # Convert to pandas dataframe for later print-out
    report_df = pd.DataFrame(report).transpose()

    return report_df


def evaluate_wrapper(metrics: list,
                     metrics_output_dir: str,
                     y_test_path: str,
                     y_pred_folder_path: str,
                     **kwargs_evaluate) -> None:
    """
    Save all the evaluation metrics to a file.

    Args:
        metrics (`list`): List of metrics to be calculated and saved
        **kwargs_evaluate (`dict`): Dictionary `save_metrics_to_file`
            defined in the config file.
            - metrics_output_dir (`str`): Directory to save the metrics.
            - metrics_output_filename (`str`): File name to save the metrics.

    Returns:
        None

    Raises:
        KeyError: If either `retrieve_ytest_ypred` or `save_metrics_to_file` is
            not defined in the config file.
        IOError, OSError: Errors when creating the metrics file
    """
    logger.info("Evaluating the model.")

    # Validate ypred_output_dir exists. If not, create a new directory at
    # ypred_output_dir.
    metrics_output_filename = kwargs_evaluate["metrics_output_filename"]
    if not os.path.exists(metrics_output_dir):
        logger.warning("Output directory does not exist: %s. \
            Creating new directory.", metrics_output_dir)
        os.makedirs(metrics_output_dir)
        logger.info("Created directory: %s",
                    metrics_output_dir)

    # Create the output file with specified name and path
    metrics_filepath = os.path.join(metrics_output_dir,
                                    metrics_output_filename
                                    + ".txt")
    try:
        metrics_file = open(metrics_filepath, "w+", encoding="utf-8")
    except (IOError, OSError) as e:
        logger.error("Error creating metrics file: %s", e)
        raise e

    # Read test data
    y_test = read_test_from_path(y_test_path)

    # # Read vectorizer saved from train setp
    # vectorizer = read_vectorizer_from_path(kwargs_evaluate["vectorizer_path"])

    # Define a target encoder
    target_encoder = LabelEncoder()

    # Iterate through each of the 4 MBTI dimensions, get y_pred, and compare with y_test
    for col in ["I", "S", "F", "J"]:
        # Read y_pred from file
        y_pred_filename = kwargs_evaluate["y_pred_filename_prefix"] + \
            "_" + col + "=1.csv"
        y_pred_path = os.path.join(y_pred_folder_path, y_pred_filename)
        y_pred = read_pred_from_path(y_pred_path)

        # Specify test target column
        y_test_col = y_test[col]

        # verify y_test and y_pred have the same length
        verify_test_and_pred(y_test_col, y_pred)

        # vectorize corresponding target column in y_test
        test_target_vector = target_encoder.fit_transform(y_test_col)

        # Save the metrics to a file
        metrics_file.write("\n--------------" + col + "=1-----------------\n")
        if "accuracy" in metrics:
            acc = accuracy_score(y_test_col, y_pred)
            try:
                metrics_file.write(f"\nAccuracy: {acc}\n")
            except TypeError:
                pass

        if "confusion_matrix" in metrics:
            confusion_df = confusion_mat(y_test_col, y_pred)
            metrics_file.write("\nConfusion matrix:\n")
            try:
                metrics_file.write(confusion_df.to_string())
            except AttributeError:
                pass
            metrics_file.write("\n")

        if "classification_report" in metrics:
            report_df = class_report(y_test_col, y_pred)
            metrics_file.write("\nClassification report:\n")
            try:
                metrics_file.write(report_df.to_string())
            except AttributeError:
                pass
            metrics_file.write("\n")

    metrics_file.close()

    logger.info("Saved model evaluation metrics to file %s.", metrics_filepath)
