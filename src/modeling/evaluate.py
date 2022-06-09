import logging
import os

import pandas as pd
from pandas.errors import ParserError
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)


def read_test_from_path(test_path: str) -> pd.DataFrame:
    """Read the test data saved in previous train_test_split from given path

    Args:
        test_path (`str`): Path to the test file

    Returns:
        A pandas DataFrame containing the test set.

    Raises:
        FileNotFoundError: If the file is not found.
        ParserError: If the file fails to parse.
    """
    logger.info("Retrieving y_test from previous train-test-split from file.")
    # Load y_test saved in previous step
    try:
        test = pd.read_csv(test_path)
        logger.info("y_test loaded from %s.", test_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", test_path)
        raise e
    except ParserError as e:
        logger.error("Error parsing data from %s", test_path)
        raise e
    except Exception as e:
        logger.error("Unknown error reading data from %s", test_path)
        raise e

    return test


def read_pred_from_path(y_pred_path: str) -> list:
    """Read the predicted classes of the test data from local file.

    Args:
        y_pred_path (`str`): Path to the y_pred file saved in previous predictions.

    Returns:
        A pandas Series containing predicted classes.

    Raises:
        FileNotFoundError: If the file is not found.
        ParserError: If the file fails to parse.
    """

    logger.info("Retrieving y_pred from previous prediction.")
    # Load y_pred saved in previous step
    try:
        y_pred = pd.read_csv(y_pred_path)
        logger.info("y_pred loaded from %s.", y_pred_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", y_pred_path)
        raise e
    except ParserError as e:
        logger.error("Error parsing data from %s", y_pred_path)
        raise e
    except Exception as e:
        logger.error("Unknown error reading data from %s", y_pred_path)
        raise e

    y_pred_bin = y_pred["class"].tolist()

    return y_pred_bin


def verify_test_and_pred(y_test: pd.Series, y_pred: list) -> None:
    """Verify if the test and prediction data have the same length.

    Args:
        y_test (`pd.Series`): Actual target classes.
        ypred_bin (`list`): Predicted target classes.

    Returns:
        None

    Raises:
        ValueError: If the test and prediction data have different length.
    """

    logger.info("Verifying test and prediction data.")
    # Verify if the test and prediction data are the same length
    if len(y_test.index) != len(y_pred):
        logger.error("Test and prediction data are not of the same length.")
        raise ValueError(
            "Test and prediction data are not of the same length.")


def confusion_mat(y_test: pd.Series, ypred_bin: list) -> pd.DataFrame:
    """Calculate the confusion matrix between the actual and predicted test target data.

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


def class_report(y_test: pd.Series, ypred_bin: list) -> pd.DataFrame:
    """Generate the classification report.

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
                     test_path: str,
                     y_pred_folder_path: str,
                     **kwargs_evaluate) -> None:
    """Wrapper function for model evaluations.

    The test set from previous train-test-split and the predictions from previous training runs
    are loaded. Evaluation metrics including confusion matrix and classifiation report are
    generated and saved to local files.

    Args:
        metrics (`list`): List of metrics to be evaluated, choices are "confusion_matrix"
            and "classification_report".
        metrics_output_dir (`str`): Path to the directory to store the metrics files.
        test_path (`str`): Path to the test file saved in previous train-test-split.
        y_pred_folder_path (`str`): Path to the directory containing prediction files from
            previous training runs.
        kwargs_evaluate (`dict`): Dictionary `evaluate_wrapper` defined in config.yaml
            - metrics_output_filename (`str`): Name of the metrics output file.
            - y_pred_filename_prefix (`str`): Prefix of the prediction file names.

    Returns:
        None

    Raises:
        IOError, OSError: Errors when creating the metrics file
    """

    logger.info("Evaluating the model.")

    # Validate ypred_output_dir exists. If not, create a new directory at
    # ypred_output_dir.
    metrics_output_filename = kwargs_evaluate["metrics_output_filename"]
    if not os.path.exists(metrics_output_dir):
        logger.warning("Output directory does not exist: %s. "
                       "Creating new directory.", metrics_output_dir)
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
    test = read_test_from_path(test_path)

    # Iterate through each of the 4 MBTI dimensions, get y_pred, and compare with y_test
    for col in ["I", "S", "F", "J"]:
        # Read y_pred from file
        y_pred_filename = kwargs_evaluate["y_pred_filename_prefix"] + \
            "_" + col + "=1.csv"
        y_pred_path = os.path.join(y_pred_folder_path, y_pred_filename)
        y_pred = read_pred_from_path(y_pred_path)

        # Specify test target column
        y_test_col = test[col]

        # verify y_test and y_pred have the same length
        verify_test_and_pred(y_test_col, y_pred)

        # Save the metrics to a file
        metrics_file.write("\n--------------" + col + "=1-----------------\n")
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
