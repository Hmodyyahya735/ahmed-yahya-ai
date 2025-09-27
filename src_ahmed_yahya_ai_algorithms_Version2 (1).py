import numpy as np

def linear_regression(X, y):
    """
    خوارزمية الانحدار الخطي البسيط باستخدام numpy
    ترجع معامل الميل والتقاطع
    """
    X = np.array(X)
    y = np.array(y)
    n = len(X)
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    numer = np.sum((X - mean_x) * (y - mean_y))
    denom = np.sum((X - mean_x) ** 2)
    a = numer / denom
    b = mean_y - a * mean_x
    return {"slope": a, "intercept": b}

def logistic_regression_predict(X, w, b):
    """
    التنبؤ باستخدام الانحدار اللوجستي (احتمالية الانتماء للفئة)
    """
    X = np.array(X)
    z = w * X + b
    return 1 / (1 + np.exp(-z))