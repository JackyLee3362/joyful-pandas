import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    # r2_score, # 决定系数，通常和回归问题相关
    # roc_curve, # 返回的数组，暂时不考虑
    # classification_report, # 一个报告，用处不大
)

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)


class FairMetric:

    def __init__(self, name, y_true, y_pred, sensitive_features: pd.Series = None):
        self.y_true = y_true
        self.name = name
        self.y_pred = y_pred
        self.sensitive_features = sensitive_features
        self.performance_metrics = pd.Series(name="Performance" + name)
        self.fairness_metrics = pd.Series(name="Fair" + name)
        self.eval_metrics()

    def eval_performance_metrics(self):
        """评估性能指标"""
        metrics = self.performance_metrics
        y_true, y_pred = self.y_true, self.y_pred
        metrics["准确度(1)"] = accuracy_score(y_true, y_pred)
        metrics["精确度(1)"] = precision_score(y_true, y_pred)
        metrics["召回率(1)"] = recall_score(y_true, y_pred)
        metrics["f1分数(1)"] = f1_score(y_true, y_pred)
        # s['回归系数(1)'] = r2_score(y_true, y_pred)
        metrics["AUC分数(1)"] = roc_auc_score(y_true, y_pred)

        # tnr, fpr, fnr, tpr = confusion_matrix(y_true, y_pred).ravel()
        # metrics["TPR(1)"] = tpr
        # metrics["TNR(1)"] = tnr
        # metrics["FPR(0)"] = fpr
        # metrics["FNR(0)"] = fnr

    def eval_fairness_metrics(self):
        """评估公平性指标"""
        if self.sensitive_features is None:
            raise ValueError("敏感属性未赋值")
        sf = self.sensitive_features
        metrics = self.fairness_metrics
        y_true, y_pred = self.y_true, self.y_pred
        metrics["DP差异(0)"] = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sf
        )
        metrics["DP比率(1)"] = demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sf
        )
        metrics["EO差异(0)"] = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sf
        )
        metrics["EO比率(1)"] = equalized_odds_ratio(
            y_true, y_pred, sensitive_features=sf
        )

    def eval_metrics(self):
        """评估性能和公平性指标"""
        self.eval_performance_metrics()
        self.eval_fairness_metrics()
        self.metrics = pd.concat([self.performance_metrics, self.fairness_metrics])
        self.metrics.name = self.name
