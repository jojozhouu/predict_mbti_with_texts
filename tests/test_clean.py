import os
import pandas as pd
import pytest

from src.modeling.clean import create_binary_target, define_stopwords, remove_punc, remove_stopwords, replace_url_to_link, uppercase_types, verify_types

raw_input = "Test input. With LINKS 'https://www.google.com/', another \
    link https://planitpurple.northwestern.edu/. different word form: go, goes, \
 going, goin, goin'. A LOT OF PUNCTUATION!?!?!?!?@#$%^&*()_+-=[]{};':,./<>? THIS \
 should be enough :) "

raw_valid = [["INTJ", "post 1 post 1 post 1"],
             ["ENFP", "post 2 post 2 post 2"],
             ["INTP", "post 3 post 3 post 3"],
             ["ENTP", "post 4 post 4 post 4"]]

raw_lower = [["intj", "post 1 post 1 post 1"],
             ["ENFP", "post 2 post 2 post 2"],
             ["intp", "post 3 post 3 post 3"],
             ["ENTP", "post 4 post 4 post 4"]]

raw_invalid = [["NOT A TYPE", "post 1 post 1 post 1"],
               ["ENFP", "post 2 post 2 post 2"],
               ["NOT A TYPE EITHER", "post 3 post 3 post 3"],
               ["ENTP", "post 4 post 4 post 4"]]


def test_replace_url() -> None:
    """
    Test whether replace_url_to_link function replaces URLs with the word `link`.
    Test passed if no error raised
    """
    cleaned_text = replace_url_to_link(raw_input)
    true_text = "Test input. With LINKS 'link another     "\
        "link link different word form: go, goes,  going, goin, goin'. "\
        "A LOT OF PUNCTUATION!?!?!?!?@#$%^&*()_+-=[]{};':,./<>? "\
        "THIS  should be enough :) "

    assert cleaned_text == true_text


def test_remove_punc() -> None:
    """
    Test whether remove_punc function removes punctuation from text.
    Test passed if no error raised
    """
    cleaned_text = remove_punc(raw_input)
    true_text = "Test input With LINKS httpswwwgooglecom another     "\
        "link httpsplanitpurplenorthwesternedu different word "\
        "form go goes  going goin goin A LOT OF PUNCTUATION "\
                "THIS  should be enough  "

    assert cleaned_text == true_text


def test_remove_punc_empty_string() -> None:
    """
    Test whether remove_punc function removes punctuation from text if string empty.
    Test passed if no error raised
    """
    cleaned_text = remove_punc(" ")
    true_text = " "

    assert cleaned_text == true_text


def test_remove_stopwords() -> None:
    """
    Test whether remove_stopwords function removes stopwords from text.
    Test passed if no error raised
    """
    sw = define_stopwords()
    cleaned_text = remove_stopwords(raw_input, sw)
    true_text = "T e   n p u .   W h   L I N K S   ' h p : / / w w w . g g l e . c / ' ,   "\
        "n h e r           l n k   h p : / / p l n p u r p l e . n r h w e e r n . "\
        "e u / .   f f e r e n   w r   f r :   g ,   g e ,     g n g ,   g n "\
                ",   g n ' .   A   L O T   O F   P U N C T U A T I O N ! ? ! ? ! "\
        "? ! ? @ # $ % ^ & * ( ) _ + - = [ ] { } ; ' : , . / < > ?   "\
        "T H I S     h u l   b e   e n u g h   : )  "

    assert cleaned_text == true_text


def test_verify_types() -> None:
    """Test whether verify_types function raises error if input
    type is not one of the 16 MBTI types."""
    df_test = pd.DataFrame(raw_invalid, columns=["type", "posts"])
    with pytest.raises(ValueError):
        verify_types(df_test)


def test_verify_types_happy() -> None:
    """Test whether verify_types function does not raise error if input
    type are all one of the 16 MBTI types."""
    df_test = pd.DataFrame(raw_valid, columns=["type", "posts"])
    verify_types(df_test)


def test_uppercasee() -> None:
    """Test whether the function convert the `type` column
    to uppercase"""
    df_test = pd.DataFrame(raw_lower, columns=["type", "posts"])
    df_out = uppercase_types(df_test)
    df_true = pd.DataFrame(raw_valid, columns=["type", "posts"])

    pd.testing.assert_frame_equal(df_true, df_out)


def test_create_binary_target() -> None:
    """Test whether the function create_binary_target()
    creates 4 binary columns"""
    df_test = pd.DataFrame(raw_valid, columns=["type", "posts"])
    df_out = create_binary_target(df_test)

    df_true = [["INTJ", "post 1 post 1 post 1", 1, 0, 0, 1],
               ["ENFP", "post 2 post 2 post 2", 0, 0, 1, 0],
               ["INTP", "post 3 post 3 post 3", 1, 0, 0, 0],
               ["ENTP", "post 4 post 4 post 4", 0, 0, 0, 0]]
    df_true = pd.DataFrame(
        df_true, columns=['type', 'posts', 'I', 'S', 'F', 'J'])

    pd.testing.assert_frame_equal(df_true, df_out)
