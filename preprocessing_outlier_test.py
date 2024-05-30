import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

df = pd.read_csv('merged_data\modified_merged_data_after_remove_outlier.csv')

# 사용일자를 datetime 형식으로 변환
df['사용일자'] = pd.to_datetime(df['사용일자'], format='%Y%m%d')

# 요일, 주말/평일, 월, 계절 변수 생성
df['요일'] = df['사용일자'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['평일'] = df['요일'].apply(lambda x: 1 if x < 5 else 0)
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5  else 0)

df = df[df['평일'] == 0]    # 평일만 남김

# Calculate mean and standard deviation
mean = np.mean(df['총승객수'])
std = np.std(df['총승객수'])

# Calculate z-scores
df['Z_Score'] = (df['총승객수'] - mean) / std

print(df)
print()

# Set a threshold (e.g., |Z| > 3)
threshold = 3

# Identify outliers
outliers = df[abs(df['Z_Score']) > threshold]

print("Outliers:")
print(outliers)

