import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tabulate import tabulate
import seaborn as sns

data = pd.read_csv('merged_data/data_after_encoding.csv')
#data = pd.read_csv('merged_data/data_after_remove_dirty_data.csv')

# 데이터 프레임에 있는 모든 column
columns = data.columns.values.tolist()

df = data.copy()

#print(tabulate(df[df['역명'] == '강남'], headers='keys', tablefmt='pretty'))

# Scaling할 때 사용하지 않을 column 선정
columns_dropped = ['사용일자', '노선명', '역명', '행정구역_경기남부', '행정구역_경기동부',
                   '행정구역_경기북서부', '행정구역_경기중부', '행정구역_서울', '행정구역_인천', '강수량', '적설']

# Scaling할 column 선정
columns_after_drop = [col for col in columns if col not in columns_dropped]

def process_rain_or_snow(df, column):
    # 비나 적설량이 있는 날과 없는 날을 나눔
    non_zero_days = df[df[column] > 0].copy()
    zero_days = df[df[column] == 0].copy()

    # 로그 변환
    non_zero_days[column] = np.log1p(non_zero_days[column])

    # RobustScaler 적용
    scaler = MinMaxScaler()
    non_zero_days[column] = scaler.fit_transform(non_zero_days[[column]])

    # 비나 적설량이 있는 날과 없는 날 합치기
    df_final = pd.concat([non_zero_days, zero_days])

    return df_final


df[['총이용승객수']] = df[['총이용승객수']].applymap(lambda x: np.log1p(x))

# 강수량과 적설량 처리
df = process_rain_or_snow(df, '강수량')
df = process_rain_or_snow(df, '적설')
print(df['강수량'].describe())
print(df)

print(tabulate(df.head(5), headers='keys', tablefmt='pretty'))

# 스케일링 하지 않은 column들의 데이터
df_not_scaled = df[columns_dropped]
print(tabulate(df_not_scaled[df_not_scaled['사용일자'] == 20231026], headers='keys', tablefmt='pretty'))

"""
# 로그 변환 후 강수량 데이터 분포 시각화
plt.figure(figsize=(12, 6))
sns.histplot(df['강수량'], kde=True)
plt.title('Log-Transformed 강수량 분포')
plt.show()
"""
"""
# Standard Scaler 이용
standard_scaler = StandardScaler()

df_standard = standard_scaler.fit_transform(df[columns_after_drop])
df_standard = pd.DataFrame(df_standard, columns=columns_after_drop)

# 스케일하지 않은 데이터와 결합
df_standard_scaled = pd.concat([df_not_scaled, df_standard], axis=1)
"""

# Robust Scaler 이용
robust_scaler = RobustScaler()

df_robust = robust_scaler.fit_transform(df[columns_after_drop])
df_robust = pd.DataFrame(df_robust, columns=columns_after_drop)

# 스케일하지 않은 데이터와 결합
df_scaled = pd.concat([df_not_scaled, df_robust], axis=1)

"""
# Min-max Scaler 이용
min_max_scaler = MinMaxScaler()

df_minmax = min_max_scaler.fit_transform(df[columns_after_drop])
df_minmax = pd.DataFrame(df_minmax, columns=columns_after_drop)

df_minmax_scaled = pd.concat([df_not_scaled, df_minmax], axis=1)
"""

#print(df_robust_scaled)
#print(tabulate(df_standard.head(5), headers='keys', tablefmt='pretty'))
#print(tabulate(df_robust.head(5), headers='keys', tablefmt='pretty'))
#print(tabulate(df_minmax.head(5), headers='keys', tablefmt='pretty'))

#print(tabulate(df_standard_scaled.head(5), headers='keys', tablefmt='pretty'))
#print(tabulate(df_robust_scaled.head(5), headers='keys', tablefmt='pretty'))
#print(tabulate(df_minmax_scaled.head(5), headers='keys', tablefmt='pretty'))

#df_standard_scaled.to_csv('merged_data/standard_scaled.csv', index=False)
#df_robust_scaled.to_csv('merged_data/data_after_scaling.csv')

print(tabulate(df_scaled[df_scaled['사용일자'] == 20231026], headers='keys', tablefmt='pretty'))

print(df_scaled)
"""
# 스케일러 인스턴스 생성
scalers = {
    'Original': df,
    'StandardScaler': StandardScaler().fit_transform(df[columns_after_drop]),
    'RobustScaler': RobustScaler().fit_transform(df[columns_after_drop]),
    'MinMaxScaler': MinMaxScaler().fit_transform(df[columns_after_drop])
}


# 데이터 분포 시각화
plt.figure(figsize=(14, 10))

for i, (name, scaled_data) in enumerate(scalers.items(), 1):
    plt.subplot(2, 2, i)
    if name == 'Original':
        sns.histplot(data=scaled_data, kde=True)
    else:
        sns.histplot(data=pd.DataFrame(scaled_data, columns=columns_after_drop), kde=True)
    plt.title(name)

plt.tight_layout()
plt.show()
"""

df_scaled.to_csv('merged_data/data_after_scaling.csv')

# 히스토그램과 KDE 플롯
def plot_distributions(df, columns, title):
    plt.figure(figsize=(14, 7))
    for col in columns:
        sns.histplot(df[col], kde=True, label=col, bins=30)
    plt.legend()
    plt.title(title)
    plt.show()

# 강수량, 적설량, 총이용승객수의 분포 시각화
plot_distributions(df_scaled, ['강수량', '적설', '총이용승객수'], '강수량, 적설량, 총이용승객수 분포')

# 특정 컬럼의 분포 시각화 예시 (강수량)
plt.figure(figsize=(12, 6))
sns.histplot(df_scaled['강수량'], kde=True)
plt.title('강수량 분포')
plt.show()

# 특정 컬럼의 분포 시각화 예시 (적설량)
plt.figure(figsize=(12, 6))
sns.histplot(df_scaled['적설'], kde=True)
plt.title('적설량 분포')
plt.show()

# 특정 컬럼의 분포 시각화 예시 (총이용승객수)
plt.figure(figsize=(12, 6))
sns.histplot(df_scaled['총이용승객수'], kde=True)
plt.title('총이용승객수 분포')
plt.show()