import re
import string
import logging
import os
from typing import Union

import pandas as pd
from pandas.errors import ParserError
import nltk
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Define global lemmatizer
wnl = WordNetLemmatizer()


def check_dl_nltk_data(**kwargs_nltk_dl) -> None:
    """
    Check if the NLTK stopwords, wordnet, omw-1.4 and punkt are downloaded.
    If not, download them.

    Args:
        kwargs_nltk_dl (`dict`): Dictionary of arguments for nltk.download().
            - download_dir (`str`): Directory to download data to.
    Returns:
        None

    Raises:
        None
    """
    logger.info("Checking if NLTK stopwords is downloaded.")

    # Add custom data folder to nltk search path
    nltk.data.path.append(os.path.join(
        os.getcwd(), kwargs_nltk_dl["download_dir"]))

    # Try to find packages in nltk data folder. If not found, download them
    # Check for stopwords
    try:
        nltk.data.find("stopwords")
    except LookupError:
        logger.info("NLTK stopwords is not downloaded. Downloading...")
        nltk.download("stopwords", kwargs_nltk_dl["download_dir"])
    else:
        logger.info("NLTK stopwords is already downloaded.")

    # Check for wordnet
    logger.info("Checking if NLTK wordnet is downloaded.")
    try:
        nltk.data.find("wordnet")
    except LookupError:
        logger.info(
            "NLTK wordnet is not downloaded. Downloading...")
        nltk.download("wordnet", kwargs_nltk_dl["download_dir"])
    else:
        logger.info("NLTK wordnet are already downloaded.")

    # Check for omw-1.4
    logger.info("Checking if NLTK omw-1.4 is downloaded.")
    try:
        nltk.data.find("omw-1.4")
    except LookupError:
        logger.info("NLTK omw-1.4 is not downloaded. Downloading...")
        nltk.download("omw-1.4", kwargs_nltk_dl["download_dir"])
    else:
        logger.info("NLTK omw-1.4 is already downloaded.")

    # Check for punkt
    logger.info("Checking if NLTK punkt is downloaded.")
    try:
        nltk.data.find("punkt")
    except LookupError:
        logger.info("NLTK punkt is not downloaded. Downloading...")
        nltk.download("punkt", kwargs_nltk_dl["download_dir"])
    else:
        logger.info("NLTK punkt is already downloaded.")


def read_raw_data(raw_data_path: str) -> pd.DataFrame:
    """Read raw data stored in the given path, return a pandas DataFrame.

    Args:
        read_raw_data (`str`): Path to the raw data with two
            columns `type` and `posts`.

    Returns:
        A pandas DataFrame containing raw data.

    Raises:
        FileNotFoundError: If the raw data file is not found.
        ParserError: If the raw data file is not in the correct format.
        TypeError: If the raw data does not end with .csv

    """
    logger.info("Reading data from %s", raw_data_path)

    # Read data from source_url.
    if raw_data_path.endswith(".csv"):
        try:
            raw_data = pd.read_csv(raw_data_path, encoding="ISO-8859-1")
        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise e
        except ParserError as e:
            logger.error("Error parsing data from %s", raw_data_path)
            raise e
        except Exception as e:
            logger.error("Unknown error reading data from %s", raw_data_path)
            raise e

    else:
        raise TypeError(f"Unsupported file extension: {raw_data_path}. "
                        "Currently only .csv files are supported")

    return raw_data


def verify_types(data: pd.DataFrame) -> pd.DataFrame:
    """Verify that the `type` column only contains one of the 16 MBTI types.

    Args:
        data (`pandas.DataFrame`): DataFrame to verify.

    Returns:
        pandas.DataFrame: DataFrame with verified MBTI types.

    Raises:
        ValueError: If the `type` column contains invalid MBTI types
    """

    logger.info("Verifying MBTI types in data.")
    # Get unique MBTI type in the data and the 16 valid types
    type_in_data = data["type"].unique()
    possible_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP",
                      "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ",
                      "ISTP", "ISFP", "ESTP", "ESFP"]

    # Check if train data only contains one of the 16 MBTI types, if not, raise error
    check = all(x in possible_types for x in type_in_data)
    if not check:
        logger.error(
            "MBTI types in data contain invalid types. Please check data.")
        raise ValueError("MBTI types in data are not valid.")

    # Check if train data have all possible types, if not, warn user
    if len(type_in_data) != len(possible_types):
        logger.warning("Not sufficient types in data! Results could be biased. "
                       "Data only have %s types", len(type_in_data))
    else:
        logger.info("Type column in data has 16 types.")

    return data


def uppercase_types(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the `type` column to uppercase.

    Args:
        data (`pandas.DataFrame`): DataFrame to convert.

    Returns:
        pandas.DataFrame: DataFrame with MBTI types in uppercase.

    Raises:
        None
    """
    # Convert type column to uppercase
    logger.info("Converting MBTI types to uppercase.")
    data["type"] = data["type"].str.upper()

    return data


def create_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the column of 4-dimension MBTI type to 4 columns of binary
    types.

    For example, `INTJ` will be converted to 4 columns, with `I`=1, `S`=0,
    `F`=0, and `J`=1.

    Args:
        data (`pandas.DataFrame`): DataFrame to create binary target columns.

    Returns:
        A pandas DataFrame with binary target columns created

    Raises:
        None
    """

    logger.debug("Creating 4 columns of binary target.")
    # Create binary target columns, for example, `INTJ` will be converted to additional
    # 4 columns, with `I`=1, `S`=0, `F`=0, and `J`=1.
    data["I"] = data["type"].apply(lambda x: 1 if "I" in x else 0)
    data["S"] = data["type"].apply(lambda x: 1 if "S" in x else 0)
    data["F"] = data["type"].apply(lambda x: 1 if "F" in x else 0)
    data["J"] = data["type"].apply(lambda x: 1 if "J" in x else 0)
    logger.debug("Successsfully created binary targets.")

    return data


def define_stopwords() -> set:
    """Define stopwords as the union of the stopwords in the NLTK wordnet and
    stopwords with punctuation removed.

    Args:
        None

    Returns:
        set: Set of stopwords.

    Raises:
        None
    """
    logger.debug("Defining stopwords.")
    # Get stopwords from NLTK wordnet
    sw_reg = nltk.corpus.stopwords.words("english")

    # Get stopwords from NLTK wordnet, but with punctuation removed
    sw_no_punc = re.sub("[" + re.escape(string.punctuation) + "]", "",
                        " ".join(sw_reg)).split()

    # Combine the two lists into a set
    stopwords = set(sw_reg + sw_no_punc)

    logger.debug("Successfully defined stopwords.")

    return stopwords


def replace_url_to_link(txt: str) -> str:
    """Replace URLs with word `link`.

    Args:
        text (`str`): Text to replace URLs.

    Returns:
        The text with URLs replaced.
    """
    txt = re.sub(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "link", txt)

    return txt


def remove_punc(txt: str) -> str:
    """Remove punctuation.

    Args:
        text (`str`): Text to remove punctuation.

    Returns:
        Text with punctuation removed.
    """
    txt = re.sub("[" + re.escape(string.punctuation) + "]", "", txt)

    return txt


def remove_stopwords(txt: str, stopwords: set) -> str:
    """Remove stopwords.

    Args:
        text (`str`): Text to remove stopwords.
        stopwords (`set`): Set of stopwords.

    Returns:
        Text with stopwords removed.
    """
    txt = " ".join([w for w in txt if w not in stopwords])

    return txt


def lemmatize_all(word: str) -> str:
    """Lemmatize a word.

    Args:
        word (`str`): Word to lemmatize.

    Returns:
        Lemmatized word.
    """
    # lemmatize a word with adjective, verb, and noun speech tags
    word = wnl.lemmatize(word, "a")
    word = wnl.lemmatize(word, "v")
    word = wnl.lemmatize(word, "n")
    return word


def save_clean_data_to_file(data: pd.DataFrame,
                            clean_data_output_dir: str,
                            **kwargs_save_clean_data: dict) -> None:
    """Save cleaned data to a csv file at the specified path.

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
        logger.warning("Output directory does not exist: %s. "
                       "Creating new directory.", clean_data_output_dir)
        os.makedirs(clean_data_output_dir)
        logger.info("Created directory: %s",
                    clean_data_output_dir)

    # Save cleaned data to file, if specified
    filepath = os.path.join(
        clean_data_output_dir,
        kwargs_save_clean_data["clean_data_output_filename"]) + ".csv"
    try:
        data.to_csv(filepath, index=False)
    except (IOError, OSError) as e:
        logger.error("Failed to save cleaned data to file: %s", filepath)
        raise e
    logger.info("Saved cleaned data to file: %s", filepath)


def save_stopwords_to_file(stopwords: set,
                           **kwargs_save_stopwords: dict) -> None:
    """
    Save stopwords to a csv file at the specified path.

    Args:
        stopwords (`set`): set of stopwords.
        **kwargs_save_stopwords (`dict`): Dictionary `save_stopwords_to_file`
            defined in config.yaml
            - stopwords_output_dir (`str`): Folder to save the stopwords.
            - stopwords_output_filename (`str`): Filename of the saved csv file.

    Returns:
        None

    Raises:
        IOError, OSError: If the output file is not accessible.
    """
    stopwords_output_dir = kwargs_save_stopwords["stopwords_output_dir"]
    logger.info("Saving stopwords to file: %s",
                stopwords_output_dir)

    # Validate stopwords_output_dir exists. If not, create a new directory at
    # stopwords_output_dir.
    if not os.path.exists(stopwords_output_dir):
        logger.warning("Output directory does not exist: %s. "
                       "Creating new directory.", stopwords_output_dir)
        os.makedirs(stopwords_output_dir)
        logger.info("Created directory: %s",
                    stopwords_output_dir)

    # Save stopwords to file if specified
    filepath = os.path.join(
        stopwords_output_dir,
        kwargs_save_stopwords["stopwords_output_filename"]) + ".csv"
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            for stopword in stopwords:
                file.write(f"{stopword}\n")
    except (IOError, OSError) as e:
        logger.error("Failed to save stopwords to file: %s", filepath)
        raise e
    logger.info("Saved stopwords to file: %s", filepath)


def clean_wrapper(raw_data: str,
                  clean_data_output_dir: str,
                  is_new_data: bool,
                  save_output: bool,
                  **kwargs_clean) -> Union[pd.DataFrame, str]:
    """Wrapper function for cleaning text.

    Raw data is read from specified path as either a string or a pandas
    dataframe. It is then processed by replacing URLs with word `link`, removing
    punctuation, removing stopwords, and lemmatizing. The cleaned data is then
    saved to a csv file at the specified path.

    Args:
        raw_data (`str`): either path to raw data file or the raw text to clean.
        clean_data_output_dir (`str`): Folder to save the cleaned data.
        is_new_data (`bool`): Whether the `raw_data` is the user input text, instead of
            a raw data file.
        save_output (`bool`): Whether to save the cleaned data to a csv file.
        **kwargs_clean (`dict`): Dictionary `clean_wrapper` defined in config.yaml

    Returns:
        A Pandas DataFrame of cleaned data, or the cleaned text.

    Raises:
        None
    """
    logger.info("Starting clean.py")

    # Read raw data as either a dataframe or a string
    if is_new_data:
        data = raw_data
    else:
        data = read_raw_data(raw_data)
        # verify `types` are valid
        data = verify_types(data)

    # Check if required nltk packages (e.g. stopwords) are installed.
    # If not, will install and re-import required packages
    check_dl_nltk_data(**kwargs_clean["check_dl_nltk_data"])

    if not is_new_data:
        # Create binary target.
        data = uppercase_types(data)
        data = create_binary_target(data)
        num_records = data.shape[0]
    else:
        num_records = 1

    # Define stopwords
    stopwords = define_stopwords()

    logger.info("Cleaning data...It may take 2-3 minutes.")
    for i in range(0, num_records):
        if not is_new_data:
            # Get raw posts from raw_data dataframe
            # validate if text is a df, if so does it have "posts" ccolumn
            text = data.at[i, "posts"]
        else:
            text = data

        # Replace URLs with word `link`.
        text = replace_url_to_link(text)

        # Remove punctuation.
        text = remove_punc(text)

        # Convert all words to lower cases
        text = re.sub(" +", " ", text).lower()

        # Lemmatize all words.
        text = list(map(lemmatize_all, text.split(" ")))

        # Remove stopwords.
        text = remove_stopwords(text, stopwords)

        if not is_new_data:
            # Update the cleaned text in the raw_data dataframe.
            data.at[i, "posts"] = text

    logger.info("Finished cleaning data. %s records cleaned.", num_records)

    # Save output if specified
    if save_output:
        # Save cleaned data to local system.
        save_clean_data_to_file(data, clean_data_output_dir,
                                **kwargs_clean["save_clean_data_to_file"])

        # Save stopwords to local system.
        save_stopwords_to_file(
            stopwords, **kwargs_clean["save_stopwords_to_file"])

    logger.debug("Successfully cleaned text.")

    if not is_new_data:
        return data
    return text
