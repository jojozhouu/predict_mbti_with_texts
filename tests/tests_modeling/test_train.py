import os
import numpy as np
import pandas as pd
import pytest

from src.modeling.train import create_fit_vectorizer, split_data


raw = [["INTJ", "post 1 post 1 post 1"],
       ["ENFP", "post 2 post 2 post 2"],
       ["INTP", "post 3 post 3 post 3"],
       ["ENTP", "post 4 post 4 post 4"],
       ["INTP", "post 5 post 5 post 5"],
       ["INTJ", "post 6 post 6 post 6"],
       ["INTP", "post 3 post 3 post 3"], ]
df_raw = pd.DataFrame(raw, columns=["type", "posts"])

raw_posts = df_raw["posts"]

sw = ["1", "2", "3", "4", "5", "6"]

kwargs_split = {"test_size": 0.2, "random_state": 100}

kwargs_split_zero = {"test_size": 0, "random_state": 100}

kwargs_tfidf_vec = {"max_features": 5000}
kwargs_tfidf_zero = {"max_features": 0}


def test_split_data() -> None:
    """
    Test whether split_data function splits data into training and testing sets.
    Test passed if no error raised
    """
    train, test = split_data(df_raw, **kwargs_split)
    # define train truth
    values = [["INTJ", "post 6 post 6 post 6"],
              ["INTP", "post 5 post 5 post 5"],
              ["ENTP", "post 4 post 4 post 4"],
              ["INTP", "post 3 post 3 post 3"],
              ["INTJ", "post 1 post 1 post 1"]]
    columns = ["type", "posts"]
    index = [5, 4, 3, 6, 0]
    train_true = pd.DataFrame(values, columns=columns, index=index)

    # define test truth
    values = [["ENFP", "post 2 post 2 post 2"],
              ["INTP", "post 3 post 3 post 3"]]
    columns = ["type", "posts"]
    index = [1, 2]
    test_true = pd.DataFrame(values, columns=columns, index=index)

    pd.testing.assert_frame_equal(train_true, train)
    pd.testing.assert_frame_equal(test_true, test)


def test_split_zero() -> None:
    """
    Test whether split_data function raises error if test_size is 0.
    Test passed if no error raised and test set contains no data.
    """
    test_true = pd.DataFrame()
    train, test = split_data(df_raw, **kwargs_split_zero)

    pd.testing.assert_frame_equal(test_true, test)


def test_vectorizer() -> None:
    """
    Test whether create_fit_vectorizer function create expected vectorizer
    """
    array_true = np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.]])

    # fit and transform raw_posts to check if vectorizer returns as expected
    vec = create_fit_vectorizer(raw_posts, sw, **kwargs_tfidf_vec)
    array_out = vec.transform(raw_posts).toarray()

    assert np.allclose(array_out, array_true)


def test_vectorizer_zero() -> None:
    """
    Test whether create_fit_vectorizer function could handle
    max_features=0. Test passed if ValueError raised.
    """
    # test invalid vectorizer
    with pytest.raises(ValueError):
        create_fit_vectorizer(raw_posts, sw, **kwargs_tfidf_zero)
