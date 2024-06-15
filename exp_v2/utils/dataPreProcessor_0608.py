# %%
"""数据集预处理类"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from rich.logging import RichHandler

# %%
class DataPreProcessor:
    """数据预处理器"""

    def __init__(
        self,
        *,
        dataset_name: str,
        data_path: str,
        sensitive_names: list[str],
        label: str,
        columns_mapper: dict[dict],
        np_seed: int,
        log_level: int,
    ) -> None:
        """处理数据集-流程编排"""
        self.dataset_name = dataset_name
        # 设置 numpy 随机数种子
        np.random.seed(np_seed)
        self.seed = np_seed
        self.sensitive_names = sensitive_names
        # 初始化日志
        self._init_logger(level=log_level)
        # 载入数据到 df
        self._load_df(data_path)
        # 设置列映射
        self.label = label
        self.columns_mapper = columns_mapper
        self._init_df_replace_mapper()
        # 处理缺失值
        self._handle_npnan()
        # dummy 化
        self._init_Xy_dummy()
        # 分割数据集
        self._split_Xy(self.X, self.y)

    def _init_logger(self, logger_name="default_dataset", level=None) -> None:
        """初始化日志"""
        if level is None:
            level = logging.INFO
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )
        self.log = logging.getLogger(logger_name)
        self.log.setLevel(level)

    def _load_df(self, data_path: str):
        """载入数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} 文件路径不存在")
        if not data_path.endswith("csv"):
            raise TypeError("文件类型错误，应该是 csv 文件")
        self.df = pd.read_csv(data_path, encoding="latin-1")
        self.log.debug(f"df 包含 {self.df.shape[0]} 行数据，{self.df.shape[1]} 列")

    def _init_df_replace_mapper(self):
        """替换数据"""
        self.df.replace("?", np.nan, inplace=True)
        for column, mapper in self.columns_mapper.items():
            self.df[column] = self.df[column].map(mapper)

    def _split_df(self):
        """划分 df 为 X 和 y"""
        label = self.label
        self.X = self.df.drop(label, axis=1)
        self.y = self.df[label]

    def _handle_npnan(self):
        """处理缺失值(todo)"""
        pass

    def _init_Xy_dummy(self):
        """one-hot 编码"""
        self.X = pd.get_dummies(self.X)

    def _split_Xy(self, X, y, test_size=0.3):
        """分割数据 = 训练集 + 测试集"""
        if self.seed is None:
            raise BaseException("self.seed 未定义")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )

    def get_split(self):
        """获得 X_train, X_test, y_train 和 y_test"""
        return self.X_train, self.X_test, self.y_train, self.y_test
