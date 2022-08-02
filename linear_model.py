"""linear model."""

import numpy
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_linear_model(
    x_val, y_val, test_frac=0.2, filename="trained_linear_model"
):
    """Trains a simple linear regression model with scikit-learn."""
    try:
        assert isinstance(x_val, numpy.ndarray), "x_val must be a Numpy array"
        assert isinstance(y_val, numpy.ndarray), "y_val must be a Numpy array"
        assert isinstance(
            test_frac, float
        ), "Test set fraction must be a floating point number"
        assert test_frac < 1.0, "Test set fraction must be between 0.0 and 1.0"
        assert test_frac > 0, "Test set fraction must be between 0.0 and 1.0"
        assert isinstance(filename, str), "Filename must be a string"
        assert (
            x_val.shape[0] == y_val.shape[0]
        ), "Row numbers of x_val and y_val data must be identical"

        # Shaping
        if len(x_val.shape) == 1:
            x_val = x_val.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        # Test/train split
        x_train, x_test, y_train, y_test = train_test_split(
            x_val, y_val, test_size=test_frac, random_state=42
        )
        # Instantiate
        model = LinearRegression()
        # Fit
        model.fit(x_train, y_train)
        # Save
        fname = filename + ".sav"
        dump(model, fname)
        # Compute scores
        r2_train = model.score(x_train, y_train)
        r2_test = model.score(x_test, y_test)
        # Return scores in a dictionary
        return {"Train-score": r2_train, "Test-score": r2_test}

    except AssertionError as msg:
        print(msg)
        return msg
