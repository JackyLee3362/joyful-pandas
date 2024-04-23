import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path

file_path = Path(__file__)
data_path = file_path.joinpath("..", "..", "..", "input", "adult.csv")

# 加载原始数据
df = pd.read_csv(data_path, encoding='latin-1')

# 数据预处理
df[df == '?'] = np.nan
for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)
X = df.drop(['income'], axis=1)
y = df['income']
y = y.map({"<=50K":0, ">50K":1})

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 将 【非数值类型的列】 数字化
categorical = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
for feature in categorical:
    le = preprocessing.LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])

# 数据缩放
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
# X_train, X_test, y_train, y_test

# 设置敏感属性
X_test["sex"].value_counts()
sex = X_test['sex']
sex = sex.map(lambda x : "男" if x > 0 else "女")
# 获得敏感属性的索引
feature_index = X.columns.get_loc("sex")