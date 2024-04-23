import numpy as np


def train_model_and_test(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    test_func,
    sensitive_feature,
    model_cls=None,
    desc=""
):
    model = model_cls().fit(X_train, y_train)

    def model_pred_func(X_test):
        y_pred = model.predict(X_test)
        return np.where(y_pred > 0.5, 1, 0)

    result = test_func(model_pred_func, X_test, y_test, sensitive_feature, desc)
    return model, result
