import pytest
from src.modeling.predict import IncorrectFilenameError, IncorrectNumberOfFilesError, validate_model_folder

model_folder_valid = "tests/tests_modeling/folder_valid"
model_folder_wrong_file_name = "tests/tests_modeling/folder_wrong_file_name"
model_folder_wrong_number_of_files = "tests/tests_modeling/folder_wrong_number"

new_text = "test test test test hahaha lol lol yea yeah yay happy no"
model_path_real = "output/model/logit_I=1.pkl"
vectorizer_path_real = "output/vectorizer/tfidf_vectorizer.pkl"


def test_validate_folder_valid() -> None:
    """
    Test whether validate_folder function runs without errors 
    if folder contains correct number of files.
    Test passed if no error raised
    """
    validate_model_folder(model_folder_valid)


def test_validate_folder_wrong_file_name() -> None:
    """
    Test whether validate_folder function raises error if folder contains 
    incorrect file name. Test passed if error raised.
    """
    with pytest.raises(IncorrectFilenameError):
        validate_model_folder(model_folder_wrong_file_name)


def test_validate_folder_wrong_number_of_files() -> None:
    """
    Test whether validate_folder function raises error if folder contains 
    incorrect number of files. Test passed if error raised.
    """
    with pytest.raises(IncorrectNumberOfFilesError):
        validate_model_folder(model_folder_wrong_number_of_files)
