import logging
import os
import pickle
from typing import Tuple
import numpy as np

from pandas.errors import ParserError
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)


# def get_posts_and_target(data: pd.DataFrame,
#                          predictor_colnames: list,
#                          target_colname: str) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Get the predictor and target columns from the data.

#     Args:
#         data (:obj:`pandas.DataFrame`): featurized dataframe to train and test model on.
#         predictor_colnames (`list`): Name of the predictor columns.
#         target_column (`str`): Name of the target column.

#     Returns:
#         A tuple of pandas DataFrame and pandas Series, containing predictor
#         columns and target column.

#     Raises:
#         KeyError: If the any of the predictor_colnames or target_colname are
#             not found in the data.

#     """
#     # Get the feature columns
#     try:
#         feature_columns = data[predictor_colnames]
#     except KeyError as e:
#         logger.error(
#             "At least one of the columns\
#                  not found in data: %s", predictor_colnames)
#         raise e

#     # Get the target column
#     try:
#         target_column = data[target_colname]
#     except KeyError as e:
#         logger.error("Column not found in data: %s", target_colname)
#         raise e

#     return feature_columns, target_column


def read_clean_data(clean_data_path: str) -> pd.DataFrame:
    """
    Read the cleaneddata.

    Args:
        clean_data_path (`str`): Path to the clean data.

    Returns:
        pandas DataFrame: Clean data.

    Raises:
        FileNotFoundError: If the clean data is not found.
        ParserError: If the clean data is not in the correct format.
    """
    # Read the clean data
    try:
        data = pd.read_csv(clean_data_path)
    except FileNotFoundError as e:
        logger.error("Clean data not found at %s", clean_data_path)
        raise e
    except ParserError as e:
        logger.error("Clean data not in correct format at %s", clean_data_path)
        raise e

    logger.info("Clean data read.")

    return data


def read_stopwords(stopwords_path: str) -> list:
    """
    Read the stopwords.

    Args:
        stopwords_path (`str`): Path to the stopwords.

    Returns:
        List of stopwords.

    Raises:
        FileNotFoundError: If the stopwords are not found.
    """
    # Read the stopwords
    try:
        stopwords = pd.read_csv(stopwords_path, header=None)
    except FileNotFoundError as e:
        logger.error("Stopwords not found at %s", stopwords_path)
        raise e

    logger.info("Stopwords read.")

    return stopwords.values.flatten().tolist()


def split_data(data: pd.DataFrame,
               **kwargs_split: dict) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and test sets.

    Args:
        feature_columns (:obj:`pandas.DataFrame`): Dataframe of predictor columns
        target_column (:obj:`pandas.Series`): Series of target column
        **kwargs_split (`dict`): Dictionary `split_data` defined in the config file

    Returns:
        A tuple of pandas DataFrame and pandas Series, containing X_train,
        X_test, y_train, and y_test.

    Raises:
        TypeError: Errors in specifying train_test_split parameters,
            such as unrecognized arguments
        ValueError: Errors in specifying train_test_split parameters,
            such as invalid values
    """
    # Validate train_test_split parameters
    if kwargs_split["test_size"] == 0:
        logger.warning("Test size is 0. No test set will be generated. ")
        kwargs_split["test_size"] = 0.000001

    # Perform train_test_split
    try:
        train, test = train_test_split(data, **kwargs_split)
    except TypeError as e:
        logger.error("Error splitting data: %s. Please check for "
                     "errors like unrecognized arguments ", e)
        raise e
    except ValueError as e:
        logger.error("Error splitting data: %s. Please check for errors like "
                     "invalid hyperparameter values ", e)
        raise e

    # if test zie is 0, then move the 1 record in test to train
    if kwargs_split["test_size"] == 0.000001:
        train = pd.concat([train, test.iloc[0:1]])
        test = pd.DataFrame()

    logger.info("Train-test split finished.")

    return train, test


def save_split_to_files(train: pd.DataFrame,
                        test: pd.DataFrame,
                        split_output_dir: str,
                        **kwargs_save_split: dict) -> None:
    """
    Save the training and test sets to files.

    Args:
        X_train (:obj:`pandas.DataFrame`): Training predictor columns.
        X_test (:obj:`pandas.DataFrame`): Test predictor columns.
        y_train (:obj:`pandas.Series`): Training target column.
        y_test (:obj:`pandas.Series`): Test target column.
        **kwargs_save_split (`dict`): Dictionary `save_split_to_files` defined
            in the config file
            - X_train_filename (`str`): Name of the training predictor file.
            - X_test_filename (`str`): Name of the test predictor file.
            - y_train_filename (`str`): Name of the training target file.
            - y_test_filename (`str`): Name of the test target file.
            - output_dir (`str`): Output directory to save the files.

    Returns:
        None

    Raises:
        IOError, OSError: If any of the output file is not accessible
    """
    # Validate output_dir exists. If not, create a new directory at
    # output_dir.
    if not os.path.exists(split_output_dir):
        logger.warning("Output directory does not exist: %s. "
                       "New directory will be created.", split_output_dir)
        os.makedirs(split_output_dir)
        logger.info("Created directory: %s",
                    split_output_dir)

    # Save files to specified path with specified names
    try:
        train.to_csv(os.path.join(
            split_output_dir,
            kwargs_save_split["train_filename"])+".csv", index=False)
        test.to_csv(os.path.join(
            split_output_dir,
            kwargs_save_split["test_filename"]) + ".csv", index=False)
    except (IOError, OSError) as e:
        logger.error("Error saving files: %s", e)
        raise e

    logger.info("Saved training and test sets to directory: %s.",
                split_output_dir)


def create_fit_vectorizer(train_posts: pd.Series, sw: list, **kwargs_tfidf_vec) -> TfidfVectorizer:
    """
    Define the TfidfVectorizer object.

    Args:
        **kwargs_tfidf_vec (`dict`): Dictionary `define_tfidf_vectorizer` defined
            in the config file
            - max_features (`int`): Maximum number of features to keep.
            - min_df (`int`): Minimum number of documents a word must appear in.
            - max_df (`float`): Maximum number of documents a word can appear in.
            - ngram_range (`tuple`): Range of ngrams to use.

    Returns:
        fitted TidfVectorizer object.

    Raises:
        TypeError: If any of the parameters are not of the correct type.
        ValueError: If any of the parameters are not of the correct value.
    """
    logger.debug("Defining TfidfVectorizer object.")
    # Validate parameters
    if not isinstance(kwargs_tfidf_vec["max_features"], int):
        logger.error("max_features must be an integer.")
        raise TypeError("max_features must be an integer.")

    if kwargs_tfidf_vec["max_features"] < 1:
        logger.error("max_features must be greater than 0")
        raise ValueError("max_features must be greater than 0.")

    # Define the TfidfVectorizer object
    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=sw)
    except TypeError as e:
        logger.error("Error creating vectorizer: %s. Please check for "
                     "errors like unrecognized arguments ", e)
        raise e
    except ValueError as e:
        logger.error("Error creating vectorizer: %s. Please check for errors like "
                     "invalid hyperparameter values ", e)
        raise e

    logger.debug("TfidfVectorizer object created.")

    # Fit the vectorizer to the training data
    logger.debug("Fitting TfidfVectorizer object on train posts")
    vectorizer.fit(train_posts)

    return vectorizer


def train_logit(train_posts: np.ndarray,
                train_target: np.ndarray,
                **kwargs_logit) -> LogisticRegression:
    """
    Define the LogisticRegression object.

    Args:
        train_posts (:obj:`numpy.ndarray`): Training predictor columns.
        train_target (:obj:`numpy.ndarray`): Training target column.

    Returns:
        fitted LogisticRegression object.

    Raises:
        ValueError: If any of the parameters are not of the correct value.
    """
    logger.debug("Defining LogisticRegression object.")
    # Validate parameters
    if kwargs_logit["C"] < 0:
        logger.warning("C must be greater than 0. Setting C to 0.1.")
        kwargs_logit["C"] = 0.1

    # Define the LogisticRegression object
    try:
        logit = LogisticRegression(**kwargs_logit)
    except TypeError as e:
        logger.error("Error creating logistic regressor: %s. Please check for "
                     "errors like unrecognized arguments ", e)
        raise e
    except ValueError as e:
        logger.error("Error creating logistic regressosr: %s. Please check for errors like "
                     "invalid hyperparameter values ", e)
        raise e

    logger.debug("Logit object created.")

    # Fit the logit to the training data
    logger.debug("Fitting Logit object on train posts")
    logit.fit(train_posts, train_target)

    return logit


def save_model_to_file(logit: LogisticRegression,
                       model_output_filename: str,
                       model_output_dir: str) -> None:
    """
    Save the model object to a pickle file.

    Args:
        rf (:obj:`LogisticRegression`):Logistic Regreession model to be saved.
        model_output_dir (`str`): Output directory to save the model.
        **kwargs_save_model (`dict`): Dictionary `save_model_to_file` defined
            - model_filename (`str`): Name of the saved model file.

    Returns:
        None

    Raises:
        IOError, OSError: If the output directory is not accessible.
    """
    # Validate model_output_dir exists. If not, create a new directory at
    # model_output_dir.
    if not os.path.exists(model_output_dir):
        logger.warning("Output directory does not exist: %s. "
                       "New directory will be created.", model_output_dir)
        os.makedirs(model_output_dir)
        logger.info("Created directory: %s",
                    model_output_dir)

    # Save model as pickle file to specified path with specified name
    filepath = os.path.join(
        model_output_dir, model_output_filename)
    try:
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(logit, f)
    except (IOError, OSError) as e:
        logger.error(
            "Error saving model as pickle file. Please check permissions. %s", e)
        raise e
    logger.info("Saved model to file: %s.pkl", filepath)


def save_vectorizer_to_file(vectorizer: TfidfVectorizer,
                            vectorizer_output_dir: str,
                            **kwargs_save_vectorizer: dict) -> None:
    """
    Save the vectorizer object to a pickle file.

    Args:
        vectorizer (:obj:`TfidfVectorizer`): TfidfVectorizer object to be saved.
        vectorizer_output_dir (`str`): Output directory to save the vectorizer.
        **kwargs_save_vectorizer (`dict`): Dictionary `save_vectorizer_to_file` defined
            - vectorizer_filename (`str`): Name of the saved vectorizer file.

    Returns:
        None

    Raises:
        IOError, OSError: If the output directory is not accessible.
    """
    vectorizer_filename = kwargs_save_vectorizer["vectorizer_filename"]

    # Validate vectorizer_output_dir exists. If not, create a new directory at
    # vectorizer_output_dir.
    if not os.path.exists(vectorizer_output_dir):
        logger.warning("Output directory does not exist: %s. \
            New directory will be created.", vectorizer_output_dir)
        os.makedirs(vectorizer_output_dir)
        logger.info("Created directory: %s",
                    vectorizer_output_dir)

    # Save vectorizer as pickle file to specified path with specified name
    filepath = os.path.join(
        vectorizer_output_dir, vectorizer_filename)
    try:
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(vectorizer, f)
    except (IOError, OSError) as e:
        logger.error(
            "Error saving vectorizer as pickle file. Please check permissions. %s", e)
        raise e
    logger.info("Saved vectorizer to file: %s.pkl", filepath)


def train_wrapper(clean_data_path: str,
                  model_output_dir: str,
                  vectorizer_output_dir: str,
                  split_output_dir: str,
                  **kwargs_train) -> None:

    logger.info("Start train.py pipeline.")
    # read clean data from given path
    clean_data = read_clean_data(clean_data_path)

    # read stopwords from given path
    stopwords = read_stopwords(
        kwargs_train["read_stopwords"]["stopwords_path"])

    # split data into train and test and save to given path
    train, test = split_data(clean_data, **kwargs_train["split_data"])
    save_split_to_files(train, test, split_output_dir, **
                        kwargs_train["save_split_to_files"])

    # create TF-IDF vectorizer and fit on training data
    posts_colname = kwargs_train["train_wrapper"]["posts_colname"]
    vectorizer = create_fit_vectorizer(
        train[posts_colname], stopwords, **kwargs_train["create_fit_vectorizer"])

    # Save vectorizer to file
    save_vectorizer_to_file(
        vectorizer, vectorizer_output_dir, **kwargs_train["save_vectorizer_to_file"])

    # transform psosts column with vectorizer
    train_posts = vectorizer.transform(train[posts_colname]).toarray()

    # For each of the 4 MBTI dimensions, create encoded target, train
    # a logistic regression model, and save to given path
    for col in ["I", "S", "F", "J"]:
        logger.info("Training model for MBTI dimension: %s", col)
        # Create encoded target column
        target_encoder = LabelEncoder()
        train_target = target_encoder.fit_transform(train[col])

        # Train logistic regression model
        logit = train_logit(train_posts, train_target, **
                            kwargs_train["train_logit"])

        # Save fitted model to file
        save_model_to_file(
            logit, f"logit_{col}=1", model_output_dir)

    logger.info("train.py pipeline completed.")
