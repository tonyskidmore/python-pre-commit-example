""" linear model """

import math
from os import path

import numpy as np
import pytest
import sklearn
from joblib import load

from linear_model import train_linear_model

# from sklearn.model_selection import train_test_split


def random_data_constructor(noise_mag=1.0):
    """
    Random data constructor utility for tests
    """
    num_points = 100
    x_val = 10 * np.random.random(size=num_points)
    y_val = 2 * x_val + 3 + 2 * noise_mag * np.random.normal(size=num_points)
    return x_val, y_val


# -------------------------------------------------------------------


def fixed_data_constructor():
    """
    Fixed data constructor utility for tests
    """
    num_points = 100
    x_val = np.linspace(1, 10, num_points)
    y_val = 2 * x_val + 3
    return x_val, y_val


# -------------------------------------------------------------------


def test_model_return_object():
    """
    Tests the returned object of the modeling function
    """
    x_val, y_val = random_data_constructor()
    scores = train_linear_model(x_val, y_val)

    # =================================
    # TEST SUITE
    # =================================
    # Check the return object type
    assert isinstance(scores, dict)
    # Check the length of the returned object
    assert len(scores) == 2
    # Check the correctness of the names of the returned dict keys
    assert "Train-score" in scores and "Test-score" in scores


# -------------------------------------------------------------------


def test_model_return_vals():
    """
    Tests for the returned values of the modeling function
    """
    x_val, y_val = random_data_constructor()
    scores = train_linear_model(x_val, y_val)

    # =================================
    # TEST SUITE
    # =================================
    # Check returned scores' type
    assert isinstance(scores["Train-score"], float)
    assert isinstance(scores["Test-score"], float)
    # Check returned scores' range
    assert scores["Train-score"] >= 0.0
    assert scores["Train-score"] <= 1.0
    assert scores["Test-score"] >= 0.0
    assert scores["Test-score"] <= 1.0


# -------------------------------------------------------------------


def test_model_save_load():
    """
    Tests for the model saving process
    """
    # Disable Access to a protected member _base of a client class
    # triggered by isinstance(...) below
    # pylint: disable=W0212
    x_val, y_val = random_data_constructor()
    filename = "testing"
    _ = train_linear_model(x_val, y_val, filename=filename)

    # =================================
    # TEST SUITE
    # =================================
    # Check the model file is created/saved in the directory
    assert path.exists("testing.sav")
    # Check that the model file can be loaded properly
    # (by type checking that it is a sklearn linear regression estimator)
    loaded_model = load("testing.sav")
    assert isinstance(
        loaded_model, sklearn.linear_model._base.LinearRegression
    )


# -------------------------------------------------------------------


def test_loaded_model_works():
    """
    Tests if the loading of the model works correctly
    """
    x_val, y_val = fixed_data_constructor()
    if len(x_val.shape) == 1:
        x_val = x_val.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    filename = "testing"
    scores = train_linear_model(x_val, y_val, filename=filename)
    loaded_model = load("testing.sav")

    # =================================
    # TEST SUITE
    # =================================
    # Check that test and train scores are perfectly equal to 1.0
    assert scores["Train-score"] == 1.0
    assert scores["Test-score"] == 1.0
    # Check that trained model predicts the y_val
    # (almost) perfectly given x_val
    # Note the use of np.testing function instead of standard 'assert'
    # To handle numerical precision issues, we should use the
    # `assert_allclose` function instead of any equality check
    np.testing.assert_allclose(y_val, loaded_model.predict(x_val))


# -------------------------------------------------------------------


def test_model_works_data_range_sign_change():
    """
    Tests for functionality with data scaled high and low
    """
    # Small-valued data
    x_val, y_val = fixed_data_constructor()
    x_val = 1.0e-9 * x_val
    y_val = 1.0e-9 * y_val
    filename = "testing"
    scores = train_linear_model(x_val, y_val, filename=filename)

    # Check that test and train scores are perfectly equal to 1.0
    assert scores["Train-score"] == 1.0
    assert scores["Test-score"] == 1.0

    # Large-valued data
    x_val, y_val = fixed_data_constructor()
    x_val = 1.0e9 * x_val
    y_val = 1.0e9 * y_val
    filename = "testing"
    scores = train_linear_model(x_val, y_val, filename=filename)

    # Check that test and train scores are perfectly equal to 1.0
    assert scores["Train-score"] == 1.0
    assert scores["Test-score"] == 1.0

    # x_val-values are flipped
    x_val, y_val = fixed_data_constructor()
    x_val = -1 * x_val
    filename = "testing"
    scores = train_linear_model(x_val, y_val, filename=filename)

    # Check that test and train scores are perfectly equal to 1.0
    assert scores["Train-score"] == 1.0
    assert scores["Test-score"] == 1.0

    # y-values are flipped
    x_val, y_val = fixed_data_constructor()
    y_val = -1 * y_val
    filename = "testing"
    scores = train_linear_model(x_val, y_val, filename=filename)

    # Check that test and train scores are perfectly equal to 1.0
    assert scores["Train-score"] == 1.0
    assert scores["Test-score"] == 1.0


# -------------------------------------------------------------------


def test_noise_impact():
    """
    Tests functionality with low and high noise data and
    expected change in the R^2 score
    """
    x_val, y_val = random_data_constructor(noise_mag=0.5)
    filename = "testing"
    scores_low_noise = train_linear_model(x_val, y_val, filename=filename)

    x_val, y_val = random_data_constructor(noise_mag=5.0)
    filename = "testing"
    scores_high_noise = train_linear_model(x_val, y_val, filename=filename)

    # Check that R^2 scores from high-noise input is less than
    # that of low-noise input
    assert scores_high_noise["Train-score"] < scores_low_noise["Train-score"]
    assert scores_high_noise["Test-score"] < scores_low_noise["Test-score"]


# -------------------------------------------------------------------


def test_additive_invariance():
    """
    Tests additive invariance
    i.e. adding constant numbers to x_val or y_val array does
    not change the model coefficients
    """
    x_val, y_val = random_data_constructor(noise_mag=0.5)
    filename = "testing"

    _ = train_linear_model(x_val, y_val, filename=filename)
    m_load = load("testing.sav")
    coeff_no_additive = float(m_load.coef_)

    x_val = x_val + 100
    _ = train_linear_model(x_val, y_val, filename=filename)
    m_load = load("testing.sav")
    coeff_x_additive = float(m_load.coef_)

    y_val = y_val - 100
    _ = train_linear_model(x_val, y_val, filename=filename)
    m_load = load("testing.sav")
    coeff_y_additive = float(m_load.coef_)

    # Check that model coefficients for default and
    # additive data are same (very close)
    # Note the use of math.isclose function
    assert math.isclose(coeff_no_additive, coeff_x_additive, rel_tol=1e-6)
    assert math.isclose(coeff_no_additive, coeff_y_additive, rel_tol=1e-6)


# -------------------------------------------------------------------


def test_wrong_input_raises_assertion():
    """
    Tests for various assertion cheks written in the modeling function
    """
    x_val, y_val = random_data_constructor()
    filename = "testing"
    train_linear_model(x_val, y_val, filename=filename)

    # =================================
    # TEST SUITE
    # =================================
    # Test that it handles the case of: x_val is a string
    msg = train_linear_model("x_val", y_val)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "x_val must be a Numpy array"
    # Test that it handles the case of: y is a string
    msg = train_linear_model(x_val, "y_val")
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "y_val must be a Numpy array"
    # Test that it handles the case of: test_frac is a string
    msg = train_linear_model(x_val, y_val, test_frac="0.2")
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be a floating point number"
    # Test that it handles the case of: test_frac is within 0.0 and 1.0
    msg = train_linear_model(x_val, y_val, test_frac=-0.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    msg = train_linear_model(x_val, y_val, test_frac=1.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    # Test that it handles the case of: filename for model save a string
    msg = train_linear_model(x_val, y_val, filename=2.0)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Filename must be a string"
    # Test that function is checking input vector shape compatibility
    x_val = x_val.reshape(10, 10)
    msg = train_linear_model(x_val, y_val, filename="testing")
    assert isinstance(msg, AssertionError)
    assert (
        msg.args[0] == "Row numbers of x_val and y_val data must be identical"
    )


# -------------------------------------------------------------------


def test_raised_exception():
    """
    Tests for raised exception with pytest.raises context manager
    """
    # ValueError
    with pytest.raises(ValueError):
        # Insert a np.nan into the X array
        x_val, y_val = random_data_constructor()
        x_val[1] = np.nan
        filename = "testing"
        train_linear_model(x_val, y_val, filename=filename)
        # Insert a np.nan into the y_val array_val
        x_val, y_val = random_data_constructor()
        y_val[1] = np.nan
        filename = "testing"
        train_linear_model(x_val, y_val, filename=filename)

    with pytest.raises(ValueError) as exception:
        # Insert a string into the x_val array
        x_val, y_val = random_data_constructor()
        x_val[1] = "A string"
        filename = "testing"
        train_linear_model(x_val, y_val, filename=filename)
        assert "could not convert string to float" in str(exception.value)
