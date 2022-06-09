import os
import numpy as np
import pandas as pd
import pytest
from src.modeling.evaluate import class_report, confusion_mat, verify_test_and_pred

from src.modeling.train import create_fit_vectorizer, split_data


y_test_df = pd.DataFrame({"y_test": [1, 0, 1, 1]})
y_pred = [0.1, 0.2, 0.3, 0.4]
y_pred_invalid = [0.1, 0.2, 0.3, 0.4, 0.5]
y_pred_bin = [0, 0, 0, 0]


def test_verify() -> None:
    """ Test the verify_test_and_pred function. Test passed if no error
    raised, when test and prediction data are the same length. """

    try:
        verify_test_and_pred(y_test_df, y_pred)
    except Exception as e:
        pytest.fail("Unexpected error: %s" % e)


def test_verify_unequal_len() -> None:
    """ Test the verify_test_and_pred function. Test failed if no error
    raised, when test and prediction data are not of the same length. """

    with pytest.raises(ValueError):
        verify_test_and_pred(y_test_df, y_pred_invalid)


def test_confusion_mat() -> None:
    """ Test the confusion_mat function. Test passed if the output
    is as expected."""
    confusion_true = pd.DataFrame([[1, 0], [3, 0]],
                                  columns=["Predicted negative",
                                           "Predicted positive"],
                                  index=["Actual negative", "Actual positive"])
    confusion_out = confusion_mat(y_test_df, y_pred_bin)

    pd.testing.assert_frame_equal(confusion_true, confusion_out)


def test_class_report() -> None:
    """ Test the class_report function. Test passed if the output
    is as expected."""
    class_report_true = pd.DataFrame([[0.25, 1., 0.4, 1.],
                                      [0., 0., 0., 3.],
                                      [0.25, 0.25, 0.25, 0.25],
                                      [0.125, 0.5, 0.2, 4.],
                                      [0.0625, 0.25, 0.1, 4.]],
                                     columns=['precision', 'recall',
                                              'f1-score', 'support'],
                                     index=['0', '1', 'accuracy', 'macro avg', 'weighted avg'])
    class_report_out = class_report(y_test_df, y_pred_bin)

    pd.testing.assert_frame_equal(class_report_true, class_report_out)
