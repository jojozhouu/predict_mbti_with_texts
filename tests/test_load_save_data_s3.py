import pytest
from src.load_save_data_s3 import parse_s3


s3_path = "s3://s3name/file.csv"
s3_path_invalid = "random_path"


def test_parse_s3() -> None:
    """
    Test whether parse_s3 function separates the bucket name and file name
    as expected. Test passed if no error raised.
    """
    bucket_out = parse_s3(s3_path)
    bucket_true = "s3name"

    assert bucket_out == bucket_true


def test_parse_s3_invalid_path() -> None:
    """
    Test whether parse_s3 function raises error if s3 path is invalid.
    Test passed if error raised.
    """
    with pytest.raises(ValueError):
        parse_s3(s3_path_invalid)
