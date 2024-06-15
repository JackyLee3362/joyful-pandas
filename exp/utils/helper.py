# 普通指标和公平性指标
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    roc_curve,
    roc_auc_score,
)


from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    mean_prediction,
)


from fairlearn.metrics import (
    false_positive_rate,
    false_negative_rate,
    count,
    selection_rate,
    MetricFrame,
)

from rich import print


#
def fairness_metrics(y_true, y_pred, sensitive_features, graph=False):
    normal_metrics = {
        # "样本数(1)": count,
        "准确率(1)": accuracy_score,  # 越大越好
        "精确率(0)": precision_score,  # 越大越好
        "召回率(1)": recall_score,
        "FPR(0)": false_positive_rate,  # 越小越好
        "FNR(0)": false_negative_rate,  # 越小越好
        "F1 Score(1)": f1_score,  # f1 分数，越接近 1 越好
        # "ROCAUC的面积(1)": roc_auc_score,  # 1 表示最佳性能 0.5 表示随机分类器
        # "选择率      ": selection_rate,
        # "公平性平均预测": mean_prediction,  # 衡量不同群体之间的平均预测值是否相等
    }

    fairness_metrics = {
        "人口平等 差异(0)": demographic_parity_difference,  # 越接近 0 越好
        "人口平等 比率(1)": demographic_parity_ratio,  # 越接近 1 越好
        "机会均等 差异(0)": equalized_odds_difference,  # 越接近 0 越好
        "机会均等 比率(1)": equalized_odds_ratio,  # 越接近 1 越好
    }

    metric_frame = MetricFrame(
        metrics=normal_metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # 画图
    if graph:
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 3],
            legend=False,
            figsize=[12, 8],
            title="普通指标",
        )

    # print(metric_frame.overall)

    params = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive_features": sensitive_features,
    }

    res = metric_frame.overall.to_dict()
    for k, v in fairness_metrics.items():
        tmp = v(**params)
        # print(k, tmp)
        res[k] = tmp
    # 收集结果
    return res


def test_model(model_pred_func, X_test, y_test, sensitive, desc):
    y_pred = model_pred_func(X_test)
    return fairness_metrics(y_test, y_pred, sensitive)

if __name__ == "__main__":
    import numpy as np
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1])
    sex = np.array([1, 0, 0, 1, 0])
    fairness_metrics(y_true, y_pred, sex)