import shap
import numpy as np
import pandas as pd
from collections import namedtuple


def get_ext_train_comp_by_k(model, X_train, y_train, feature_index, k):
    """给定 k，获得 trian 拓展集合

    Args:
        model (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        feature_index (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 定义返回值
    Result = namedtuple(
        "Result",
        ["X_train_top", "y_train_top", "X_train_rand", "y_train_rand", "shap_values"],
    )
    # 计算 shap_values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    # 获得 k 个最大的索引
    arr = shap_values[:, feature_index].values  # numpy.ndarray
    sorted_indices = np.argsort(np.abs(arr))
    # 生成 top-k 个
    top_k_indices = sorted_indices[-k:]
    # 生成 rand-k 个
    random_k_index = np.random.randint(0, len(X_train), k)
    # top_k_indices

    X_train_top, y_train_top = _get_ext_train_set(X_train, y_train, top_k_indices)
    X_train_rand, y_train_rand = _get_ext_train_set(X_train, y_train, random_k_index)
    return Result(X_train_top, y_train_top, X_train_rand, y_train_rand, shap_values)
    # return (X_train_top, y_train_top), (X_train_rand, y_train_rand), shap_values


def _get_ext_train_set(X_train, y_train, indices):
    # 获得 X_ext 拓展
    X_ext = X_train.iloc[indices, :]
    y_ext = y_train.iloc[indices]
    # 敏感属性 反转
    tmp = X_train["sex"].unique()
    X_ext.loc[:, "sex"] = np.where(X_ext.loc[:, "sex"] < 0, tmp[1], tmp[0])
    # X_ext.head()
    # 索引连接
    # print(X_train.shape, X_test.shape)
    X_train_2 = pd.concat([X_train, X_ext])
    y_train_2 = pd.concat([y_train, y_ext])
    return X_train_2, y_train_2
