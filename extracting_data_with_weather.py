import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tabulate import tabulate

df = pd.read_csv('merged_data/final_data.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)

condition = (
    (df['폭염여부'] != 0) |
    (df['황사관측'] != 0) |
    (df['한파특보'] != 0) |
    (df['미세먼지'] != 0) |
    (df['초미세먼지'] != 0) |
    (df['강수량'] != 0) |
    (df['적설'] != 0)
)

df = df.loc[condition]

print(df)


# 한글 폰트 설정 (예: 나눔고딕)
font_path = '../Library/Fonts/KoPubWorld Dotum Medium.ttf'
fontprop = fm.FontProperties(fname=font_path, size=10)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# 상관관계 행렬 계산
correlation_matrix = df.corr()

# NaN 값을 0으로 대체
correlation_matrix = correlation_matrix.fillna(0)

# 상관관계 행렬 출력
print(correlation_matrix)


# 상관관계 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix', fontproperties=fontprop)

plt.show()

df.to_csv("merged_data/final_data_with_weather.csv")
