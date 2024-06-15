# %%
"""数据集预处理类"""

from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging
from rich.logging import RichHandler

Group = namedtuple("Group", ["scale_name", "origin_name", "X", "y"])


# %%
class DataPreProcessor:
    """数据预处理器"""

    def __init__(
        self,
        *,
        data_path: str = "",
        sensitive_names: list[str] = None,
        label: str = None,
        label_mapper: str = None,
        np_seed: int = 42,
        level: int = None,
    ) -> None:
        """初始化数据预处理器"""
        # 初始化随机数种子
        np.random.seed(np_seed)
        self.seed = np_seed
        self.sensi_names = sensitive_names
        # 初始化日志
        self.init_logger(level=level)
        # 初始化 df
        self.init_df(data_path)
        # 初始化数据集
        self.init_Xy(label, label_mapper)
        # 处理数据集，标签化 + 数据缩放
        self.label_X()
        self.scaler_X()
        self.group_Xy()
        self.log.info("数据集处理完毕， 遍历 xxx.grouped 查看数据，每个单元是Group(scale_name, origin_name , X, y)")

    def init_logger(self, logger_name="default_dataset", level=None) -> None:
        """初始化日志"""
        if level is None:
            level = logging.INFO
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )
        self.log = logging.getLogger(logger_name)
        self.log.setLevel(level)

    def init_df(self, data_path: str):
        """初始化数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} 文件路径不存在")
        if not data_path.endswith("csv"):
            raise TypeError("文件类型错误，应该是 csv 文件")
        self.df = pd.read_csv(data_path, encoding="latin-1")
        self.log.debug(f"df 包含 {self.df.shape[0]} 行数据，{self.df.shape[1]} 列")

    def init_Xy(self, label, mapper):
        """清洗数据"""
        self.X = self.df.drop(label, axis=1)
        self.y = self.df[label].map(mapper)
        self.X.replace("?", np.nan)
        self.split_Xy()
        self.label_X()

    def split_Xy(self, test_size=0.3):
        """分割数据为训练集和测试集"""
        if self.seed is None:
            raise BaseException("self.seed 未定义")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.seed
        )

    def label_X(self):
        """为数据集中的 object 列标签"""
        X_object_columns = self.X.select_dtypes("object").columns
        self.X_train_label = self.X_train.copy()
        self.X_test_label = self.X_test.copy()
        for feature in X_object_columns:
            le = preprocessing.LabelEncoder()
            self.X_train_label[feature] = le.fit_transform(self.X_train[feature])
            self.X_test_label[feature] = le.transform(self.X_test[feature])

    def scaler_X(self):
        """数据缩放"""
        if self.sensi_names is None:
            raise BaseException("self.sensi_names 未定义")
        # 获得原始名字
        origin_names = self.X_train[self.sensi_names].value_counts().index

        X_columns = self.X_train.columns
        # 数据缩放（需要加上 columns 和 index 参数，这样才可以保证和原来的一样）
        scaler = StandardScaler()
        self.X_train_label_scale = pd.DataFrame(
            scaler.fit_transform(self.X_train_label),
            columns=X_columns,
            index=self.X_train.index,
        )
        self.X_test_label_scale = pd.DataFrame(
            scaler.transform(self.X_test_label),
            columns=X_columns,
            index=self.X_test.index,
        )
        # 映射 原始名 -> 数字
        scale_names = self.X_train_label_scale[self.sensi_names].value_counts().index
        self.mapper_scale_origin = {}
        for key, value in zip(scale_names, origin_names):
            self.mapper_scale_origin[key] = value

    def group_Xy(self):
        result = []
        X, y = self.X_train_label_scale, self.y_train
        grouped = X.groupby(self.sensi_names)
        print(self.mapper_scale_origin)
        for name, data in grouped:
            origin = self.mapper_scale_origin[name]
            label = y[data.index]
            result.append(Group(name, origin, data, label))
        self.groups = result


if __name__ == "__main__":
    processor = DataPreProcessor(
        data_path="../../input/adult.csv",
        sensitive_names=["sex"],
        label="income",
        label_mapper={"<=50K": 0, ">50K": 1},
    )

# %%

# %%
