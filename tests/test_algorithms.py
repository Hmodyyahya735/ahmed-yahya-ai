from ahmed_yahya_ai.algorithms import linear_regression, logistic_regression_predict

def test_linear_regression():
    X = [1, 2, 3, 4]
    y = [2, 4, 6, 8]
    result = linear_regression(X, y)
    assert abs(result["slope"] - 2) < 1e-6
    assert abs(result["intercept"]) < 1e-6

def test_logistic_regression_predict():
    X = [0, 1]
    w = 1.0
    b = 0.0
    probs = logistic_regression_predict(X, w, b)
    assert abs(probs[0] - 0.5) < 1e-6
    assert 0.73 < probs[1] < 0.74
