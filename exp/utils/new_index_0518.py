import numpy as np
from tqdm import tqdm

def shapley_distance_between_individual_and_individual(
    xi: np.ndarray, xj: np.ndarray, A: np.ndarray
) -> int:
    """根据 2 个样本的 Shapley Value 计算
    可解释距离
    具体计算公式是
    $\sum_{f\in F}^f|\phi_i^f-\phi_j^f|$

    Args:
        xi (np.ndarray): 样本 xi 的 Shapley Value 的向量
        xj (np.ndarray): 样本 yi 的 Shapley Value 的向量
        A (np.ndarray): 敏感属性的 index 集合
    Return:
        int: 返回的是结果
    """
    if xi.shape != xj.shape:
        # todo Exception 应该更具体些
        raise BaseException("xi 和 xj 的格式不匹配")
    try:
        res = np.sum(np.abs(xi[A], xj[A]))
    except Exception:
        raise BaseException("敏感属性 A 的 index 超过 xi")
    return res


def shapley_distance_between_individual_and_group(
    xi_index: int,
    X: np.ndarray,
    A: np.ndarray,
) -> int:
    """根据 1 个样本的 Shapley Value 计算
    其所在群组的平均距离

    Args:
        xi_index (int): 样本 xi 的 Shapley Value 向量
        X (np.ndarray): 样本 xi 所在的敏感属性群体（不包含 xi）的 Shapley Value 矩阵
        A (np.ndarray): 敏感属性的 index 集合
    Return:
        int: 计算群体的可解释公平性距离
    """
    res = 0
    xi = X[xi_index]
    for j, xj in enumerate(X):
        if j == xi_index:
            continue
        res += shapley_distance_between_individual_and_individual(xi, xj, A)
    res /= len(X) - 1
    return res


def shapley_distance_between_group_and_group(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """计算组内的 shapley distance

    Args:
        X (np.ndarray): 同个敏感属性的组的 shapley value
        A (np.ndarray): 敏感属性的 index

    Returns:
        np.ndarray: 计算 shapley distance
    """
    total = X.shape[0]
    res = np.zeros(total)
    for i in tqdm(range(total)):
        res[i] = shapley_distance_between_individual_and_group(i, X, A)
    return res
