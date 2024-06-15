import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)


class ShapFairGroup:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shapley_values: pd.DataFrame,
        sensitive_names: list[str],
    ):

        if X.shape != shapley_values.shape:
            raise BaseException("X 和 shapley values 的 shape 不匹配")

        self.X = X

        self.y = y

        self.shapley_values = shapley_values

        self.sensitive_names = sensitive_names
        self.df_sdgg = pd.DataFrame(columns=sensitive_names)

    def sdii(self): ...

    def sdig(self, xi_idx):
        sv = self.shapley_values
        sv_feat_i = sv.loc[xi_idx, self.sensitive_names]
        sv_feat_group = sv.loc[:, self.sensitive_names]
        result = np.abs(sv_feat_i - sv_feat_group).sum(axis=0)
        self.df_sdgg.loc[xi_idx] = result / len(sv_feat_group)

    def sdgg(self, path: str = None):
        if path is not None and os.path.exists(path):
            self.df_sdgg = pd.read_csv(path, index_col=0)
            print("读取缓存: ", path)
            return
        indexes = self.shapley_values.index
        for idx in tqdm(indexes):
            self.sdig(idx)
        self.df_sdgg.to_csv(path, index=True)

    def show_sigg_hist(self, sensitive_name):
        """画统计图"""
        df = self.df_sdgg
        values, bins, bars = plt.hist(df[sensitive_name], edgecolor="white")
        plt.bar_label(bars, fontsize=10, color="navy")
        plt.margins(x=0.01, y=0.1)
        plt.show()

    def get_candidates_fair_X(self, sensitive_name, threshold):
        """获得候选集和公平样本"""
        df = self.df_sdgg
        candidates_idx = df[df[sensitive_name] > threshold].index
        fair_idx = df[df[sensitive_name] <= threshold].index
        self.X_fair = self.X.loc[fair_idx]
        self.X_candidates = self.X.loc[candidates_idx]
        print(
            f"统计: 大于 threshold {len(self.X_candidates)} 个, 小于 {len(self.X_fair)} 个"
        )

    def get_df_neighbors_by_knn(self, knn_k=7):
        knn = NearestNeighbors(n_neighbors=knn_k)
        knn.fit(self.X_fair)
        # 此处的 idx 应该是 X_fair 的序号
        distance, idxs = knn.kneighbors(self.X_candidates)
        self.idxs = idxs  # 方便调试
        self.df_neighbors = pd.DataFrame(idxs, index=self.X_candidates.index)
        # 对每行进行处理，映射到 y.iloc[row].index 或者 X.
        self.df_neighbors = self.df_neighbors.apply(
            lambda row: self.X_fair.iloc[row].index.values
        )

    def get_unfair_idx(self):
        """获取不公平的标签"""
        unfair_idx = []
        ne = self.df_neighbors
        for i in ne.index:
            neighbors = ne.loc[i].values
            # 候选者标签
            candidate_label = self.y.loc[i]
            # 公平者中的众数标签
            neighbors_mode_label = self.y.loc[neighbors].mode().values[0]

            if candidate_label != neighbors_mode_label:
                unfair_idx.append(i)

        self.unfair_idx = pd.Series(unfair_idx)
        print(
            f"候选者 {len(ne)} 个, 不公平者 {len(unfair_idx)} 个, 比例 {len(unfair_idx) / len(ne) * 100:.2f}%"
        )
