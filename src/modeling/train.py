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


def read_clean_data(clean_data_path: str) -> pd.DataFrame:
    """Read the cleaned data from local system.

    Args:
        clean_data_path (`str`): Path to the clean data.

    Returns:
        pandas DataFrame: Clean data.

    Raises:
        FileNotFoundError: If the clean data is not found.
        ParserError: If the clean data is not in the correct format.
    """
    # Read the clean data from given path
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
    """Read the stopwords from the local system.

    Args:
        stopwords_path (`str`): Path to the stopwords.

    Returns:
        List of stopwords.

    Raises:
        FileNotFoundError: If the stopwords file is not found.
    """
    # Read the stopwords from given path
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
    """Split the data into training and test sets.

    Args:
        data (:obj:`pandas.DataFrame`): Data to split.
        kwargs_split (`dict`): Dictionary `split_data` defined in the config.yaml
            - test_size (`float`): Percentage of data to use for testing.
            - random_state (`int`): Random state for the split.

    Returns:
        A tuple of pandas DataFrame, containing the train and the test samples

    Raises:
        TypeError: Errors in specifying train_test_split parameters,
            such as unrecognized arguments
        ValueError: Errors in specifying train_test_split parameters,
            such as invalid values
    """
    # Validate train_test_split parameters, if test_size is given as 0,
    # set it to 0.000001 to avoid errors. This will still assign 1 record
    # to test, which will be manually oved to the train set later.
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
    """Save the training and test sets to files.

    Args:
        train (:obj:`pandas.DataFrame`): Training set.
        test (:obj:`pandas.DataFrame`): Test set.
        split_output_dir (`str`): Path to save the train and test sets.
        kwargs_save_split (`dict`): Dictionary `save_split_to_files` defined in the config.yaml
            - train_filename (`str`): Name of the train file.
            - test_filename (`str`): Name of the test file.

    Returns:
        None

    Raises:
        IOError, OSError: raisied if any of the output file is not accessible
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


def create_fit_vectorizer(train_posts: pd.Series, sw: list,
                          **kwargs_tfidf_vec) -> TfidfVectorizer:
    """Define the TfidfVectorizer object and fit it to the training set.

    Args:
        train_posts (:obj:`pandas.Series`): Training set to fit the vectorizer on
        sw (`list`): List of stopwords.
        kwargs_tfidf_vec (`dict`): Dictionary `create_fit_vectorizer` defined in the config.yaml
            - max_features (`int`): Maximum number of features to use.

    Returns:
        fitted TidfVectorizer object.

    Raises:
        TypeError: If max_feature is not an integer.
        ValueError: If max_feature is not positive.
    """

    logger.debug("Defining TfidfVectorizer object.")
    # Validate parameters, max_features must be an integer
    if not isinstance(kwargs_tfidf_vec["max_features"], int):
        logger.error("max_features must be an integer.")
        raise TypeError("max_features must be an integer.")

    # Validate parameters, max_features must be positive
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


def check_vectorizer(vectorizer: TfidfVectorizer) -> bool:
    """Test whether the vectorizer is usable, by checking if it could transform 
    a string without raising RecursionError error

    Args:
        vectorizer (:obj:`TfidfVectorizer`): Vectorizer to test

    Returns:
        True if the vectorizer is usable, False otherwise

    Raises:
        None
    """
    logger.info("Testing vectorizer...")
    try:
        vectorizer.transform(["test test"])
    except RecursionError as e:
        logger.error(
            "Vectorizer is not usable. Creating a new vectorizer...")
        return False
    logger.info("Vectorizer is usable. Continue with training.")
    return True


def train_logit(train_posts: np.ndarray,
                train_target: np.ndarray,
                **kwargs_logit) -> LogisticRegression:
    """Define the LogisticRegression object and fit it to the training set.

    Args:
        train_posts (:obj:`numpy.ndarray`): Training predictor columns.
        train_target (:obj:`numpy.ndarray`): Training target column.
        kwargs_logit (`dict`): Dictionary `train_logit` defined in the config.yaml
            - C (`float`): Inverse of regularization strength; must be a positive float.

    Returns:
        fitted LogisticRegression object.

    Raises:
        TypeError: 
            - Errors in specifying LogisticRegression() parameters, such as 
                unrecognized arguments
            - if C is not a positive float
        ValueError: Errors in specifying LogisticRegression() parameters,
            such as invalid values 
    """
    logger.debug("Defining LogisticRegression object.")
    # Validate parameters, C must be a positive float
    if kwargs_logit["C"] < 0:
        logger.warning("C must be greater than 0. Setting C to 0.1.")
        kwargs_logit["C"] = 0.1
    if not isinstance(kwargs_logit["C"], float):
        logger.error("C must be a float.")
        raise TypeError("C must be a float.")

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
    """Save the model object to a pickle file.

    Args:
        logit (:obj:`LogisticRegression`): fitted LogisticRegression object.
        model_output_filename (`str`): Name of the model file to be saved.
        model_output_dir (`str`): Path to save the model file.

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
    """Save the vectorizer object to a pickle file.

    Args:
        vectorizer (:obj:`TfidfVectorizer`): fitted TfidfVectorizer object to be saved.
        vectorizer_output_dir (`str`): Output directory to save the vectorizer.
        kwargs_save_vectorizer (`dict`): Dictionary `save_vectorizer_to_file` defined in config.yaml
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
    """Wrapper function for the training process.

    Cleaned data is read from the specified path. It is then split into
    training and test sets. The vectorizer is then fit to the training set, and 
    the training set is transformed to be fed into the logistic model.
    1 model is created for each MBTI dimension. So, there will be 4 models in
    total that are saved in the specified path.

    Args:
        clean_data_path (`str`): Path to the cleaned data.
        model_output_dir (`str`): Path to save the models.
        vectorizer_output_dir (`str`): Path to save the vectorizer.
        split_output_dir (`str`): Path to save the split data.
        kwargs_train (`dict`): Dictionary `train` defined in config.yaml

    Returns:
        None

    Raises:
        None
    """

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

    # check if vectorizer is usable and create a new one if not, stop anyways if
    # 5 vectorizers are created and the issue persists
    iter = 0
    while (not check_vectorizer(vectorizer)) and (iter <= 4):
        vectorizer = create_fit_vectorizer(
            train[posts_colname], stopwords, **kwargs_train["create_fit_vectorizer"])
        iter += 1
    if iter > 4:
        logger.error("Vectorizer is not usable. It will raise RecursionError when transforming data. "
                     "Please check the data.")
        raise ValueError("Vectorizer is not usable. Please check the data.")

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
