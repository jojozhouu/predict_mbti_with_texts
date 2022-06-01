import re
import string
from nltk import word_tokenize
import logging
import os
from typing import Tuple, Union

import pandas as pd
from pandas.errors import ParserError
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)


def check_dl_nltk_data(**kwargs_nltk_dl) -> None:
    """
    Check if the NLTK stopwords and wordnet are downloaded. If not, download it.
    """
    logger.info("Checking if NLTK stopwords is downloaded.")
    try:
        nltk.data.find('stopwords')
    except LookupError:
        logger.info("NLTK stopwords is not downloaded. Downloading...")
        nltk.download('stopwords', **kwargs_nltk_dl)
    else:
        logger.info("NLTK stopwords is already downloaded.")

    logger.info("Checking if NLTK wordnet is downloaded.")
    try:
        nltk.data.find('wordnet')
    except LookupError:
        logger.info("NLTK wordnet is not downloaded. Downloading...")
        nltk.download('wordnet', **kwargs_nltk_dl)
    else:
        logger.info("NLTK wordnet is already downloaded.")

    # logger.info("Importing NLTK wordnet and stopwords...")
    # from nltk.corpus import stopwords
    # from nltk import wordnet  # is this necessary?????????
    # logger.info("Successfully imported NLTK wordnet and stopwords")


def read_raw_data(raw_data_path: str) -> pd.DataFrame:
    """
    Read raw data stored in local system, return a pandas DataFrame.

    Args:
        read_raw_data (`str`): Path to the raw data.

    Returns:
        pandas DataFrame of raw data.

    Raises:
        FileNotFoundError: If the raw data file is not found.
        ParserError: If the raw data file is not in the correct format.
        IOError, OSError: If the raw data file is accessible.
        TypeError: If the raw data does not end with .csv

    """
    logger.info("Reading data from %s", raw_data_path)

    # Read data from source_url.
    if raw_data_path.endswith(".csv"):
        try:
            raw_data = pd.read_csv(raw_data_path, encoding="ISO-8859-1")
        except FileNotFoundError as fe:
            logger.error("File not found: %s", fe)
            raise fe
        except ParserError as pe:
            logger.error("Error parsing data from %s", raw_data_path)
            raise pe
        except Exception as e:
            logger.error("Unknown error reading data from %s", raw_data_path)
            raise e

    else:
        raise TypeError(f"Unsupported file extension: {raw_data_path}. "
                        "Currently only .csv files are supported")

    return raw_data


def create_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the column of 4-dimension MBTI type to 4 columns of binary
    types. 

    For example, `INTJ` will be converted to 4 columns, with `I`=1, `S`=0, 
    `F`=0, and `J`=1. 
    """
    logger.info("Creating 4 columns of binary target.")
    data["I"] = data["type"].apply(lambda x: 1 if "I" in x else 0)
    data["S"] = data["type"].apply(lambda x: 1 if "S" in x else 0)
    data["F"] = data["type"].apply(lambda x: 1 if "F" in x else 0)
    data["J"] = data["type"].apply(lambda x: 1 if "J" in x else 0)
    logger.debug("Successsfully created binary targets.")

    return data


def define_stopwords() -> set:
    """
    Define stopwords.

    Returns:
        set: Set of stopwords.
    """
    logger.debug("Defining stopwords.")
    sw_reg = nltk.corpus.stopwords.words('english')
    sw_no_punc = re.sub('[' + re.escape(string.punctuation) + ']', '',
                        ' '.join(sw1)).split()
    sw = set(sw_reg + sw_no_punc)
    logger.debug("Successfully defined stopwords.")

    return sw


def replace_url_to_link(txt: str) -> str:
    """
    Replace URLs with word `link`.

    Args:
        text (`str`): Text to replace URLs.

    Returns:
        str: Text with URLs replaced.
    """
    logger.debug("Replacing URLs with word `link`.")
    txt = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "link", txt)
    logger.debug("Successfully replaced URLs with word `link`.")

    return txt


def remove_punc(txt: str) -> str:
    """
    Remove punctuation.

    Args:
        text (`str`): Text to remove punctuation.

    Returns:
        str: Text with punctuation removed.
    """
    logger.debug("Removing punctuation.")
    txt = re.sub('[' + re.escape(string.punctuation) + ']', '', txt)
    logger.debug("Successfully removed punctuation.")

    return txt


def remove_stopwords(txt: str, sw: set) -> str:
    """
    Remove stopwords.

    Args:
        text (`str`): Text to remove stopwords.
        sw (`set`): Set of stopwords.

    Returns:
        str: Text with stopwords removed.
    """
    logger.debug("Removing stopwords.")
    txt = ' '.join([w for w in word_tokenize(txt) if w not in sw])
    logger.debug("Successfully removed stopwords.")

    return txt


def lemmatize_all(word: str) -> str:
    """
    Lemmatize a word.

    Args:
        word (`str`): Word to lemmatize.

    Returns:
        str: Lemmatized word.
    """
    logger.debug("Lemmatizing word.")
    wnl = WordNetLemmatizer()
    word = wnl.lemmatize(word, 'a')
    word = wnl.lemmatize(word, 'v')
    word = wnl.lemmatize(word, 'n')
    logger.debug("Successfully lemmatized word.")

    return word


def clean_wrapper(data: pd.DataFrame, stopwords: list) -> pd.DataFrame:
    """
    Wrapper function for cleaning text.

    Args:
        data (`pandas.DataFrame`): raw data to clean.
        stopwords (`list`): List of stopwords.
    Returns:
        str: Cleaned text.
    """
    logger.info("Cleaning text for each record.")

    for i in range(0, data.shape[0]):
        logger.debug("Cleaning posts for record %d out of %d",
                     i+1, data.shape[0])
        # Get raw posts from raw_data dataframe
        text = data["posts"].iloc[i]

        # Replace URLs with word `link`.
        text = replace_url_to_link(text)

        # Remove punctuation.
        text = remove_punc(text)

        # Convert all words to lower cases
        text = re.sub(' +', ' ', text).lower()

        # Lemmatize all words.
        text = list(map(lemmatize_all, text.split(" ")))

        # Remove stopwords.
        text = remove_stopwords(text, stopwords)

        # Join all words back together.
        text = " ".join(text)

        # Update the cleaned text in the raw_data dataframe.
        data["posts"].iloc[i] = text

    logger.debug("Successfully cleaned text.")

    return data


def save_clean_data_to_file(data: pd.DataFrame,
                            clean_data_output_dir: str,
                            **kwargs_save_clean_data: dict) -> None:
    """
    Save cleaned data to a csv file at the specified path.

    Args:
        data_class (:obj:`pandas.DataFrame`): cleaned dataframe.
        clean_data_output_dir (`str`): Folder to save the cleaned data.
        **kwargs_save_clean_data (`dict`): Dictionary `save_clean_data_to_file`
            defined in config.yaml
            - clean_data_output_filename (`str`): Filename of the saved csv file.

    Returns:
        None

    Raises:
        IOError, OSError: If the output file is not accessible.
    """
    logger.info("Saving cleaned data to file: %s",
                clean_data_output_dir)

    # Validate cleaned_data_output_dir exists. If not, create a new directory at
    # cleaned_data_output_dir.
    if not os.path.exists(clean_data_output_dir):
        logger.warning("Output directory does not exist: %s. \
            Creating new directory.", clean_data_output_dir)
        os.makedirs(clean_data_output_dir)
        logger.info("Created directory: %s",
                    clean_data_output_dir)

    # Save cleaned data to file
    filepath = os.path.join(
        clean_data_output_dir,
        kwargs_save_clean_data["clean_data_output_filename"]) + ".csv"
    try:
        data.to_csv(filepath, index=False)
    except (IOError, OSError) as e:
        logger.error("Failed to save cleaned data to file: %s", filepath)
        raise e
    logger.info("Saved cleaned data to file: %s", filepath)


def save_stopwords_to_file(stopwords: list,
                           stopwords_output_dir: str,
                           **kwargs_save_stopwords: dict) -> None:
    """
    Save stopwords to a csv file at the specified path.

    Args:
        stopwords (`list`): List of stopwords.
        stopwords_output_dir (`str`): Folder to save the stopwords.
        **kwargs_save_stopwords (`dict`): Dictionary `save_stopwords_to_file`
            defined in config.yaml
            - stopwords_output_filename (`str`): Filename of the saved csv file.

    Returns:
        None

    Raises:
        IOError, OSError: If the output file is not accessible.
    """
    logger.info("Saving stopwords to file: %s",
                stopwords_output_dir)

    # Validate stopwords_output_dir exists. If not, create a new directory at
    # stopwords_output_dir.
    if not os.path.exists(stopwords_output_dir):
        logger.warning("Output directory does not exist: %s. \
            Creating new directory.", stopwords_output_dir)
        os.makedirs(stopwords_output_dir)
        logger.info("Created directory: %s",
                    stopwords_output_dir)

    # Save stopwords to file
    filepath = os.path.join(
        stopwords_output_dir,
        kwargs_save_stopwords["stopwords_output_filename"]) + ".csv"
    try:
        with open(filepath, 'w') as f:
            for stopword in stopwords:
                f.write("%s\n" % stopword)
    except (IOError, OSError) as e:
        logger.error("Failed to save stopwords to file: %s", filepath)
        raise e
    logger.info("Saved stopwords to file: %s", filepath)
