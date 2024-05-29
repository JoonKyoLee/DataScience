import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from tabulate import tabulate
import seaborn as sns

data = pd.read_csv('merged_data/data_after_encoding.csv')
#data = pd.read_csv('merged_data/data_after_remove_dirty_data.csv')

df = data.copy()

df = df.drop(columns=['역명', '노선명'])

# '사용일자'와 '행정구역'을 기준으로 평균 이용 승객 수 계산
average_ridership = df.groupby(['사용일자', '행정구역'])['총이용승객수'].mean().reset_index()

# 컬럼 이름 변경
average_ridership.rename(columns={'총이용승객수': '평균이용승객수'}, inplace=True)

# 원래 데이터프레임과 병합
df = pd.merge(df, average_ridership, on=['사용일자', '행정구역'], how='left')

# '총이용승객수' 컬럼 삭제 및 중복 제거
df = df.drop(columns=['총이용승객수'])
df = df.drop_duplicates(subset=['사용일자', '행정구역'])

df[['평균이용승객수']] = df[['평균이용승객수']].applymap(lambda x: np.log1p(x))
print(tabulate(df[df['사용일자'] == 20230401], headers='keys', tablefmt='pretty'))
print(df)

# Check for NaN values in the merged dataframe
print("Number of NaN values after merge:")
print(df.isna().sum())


# 인덱스 재설정
df = df.reset_index(drop=True)  # 이 부분 추가

# 데이터 프레임에 있는 모든 column
columns = df.columns.values.tolist()

# Scaling할 때 사용하지 않을 column 선정
columns_dropped = ['사용일자', '강수량', '적설', '행정구역']

# Scaling할 column 선정
columns_after_drop = [col for col in columns if col not in columns_dropped]

# 스케일링 하지 않은 column들의 데이터
df_not_scaled = df[columns_dropped]
print(tabulate(df_not_scaled[df_not_scaled['사용일자'] == 20231026], headers='keys', tablefmt='pretty'))


# Robust Scaler 이용
robust_scaler = RobustScaler()

df_robust = robust_scaler.fit_transform(df[columns_after_drop])
df_robust = pd.DataFrame(df_robust, columns=columns_after_drop)
print("여기")
print(tabulate(df_robust.head(6), headers='keys', tablefmt='pretty'))

# 스케일하지 않은 데이터와 결합
df = pd.concat([df_not_scaled, df_robust], axis=1)
print('8888888888')
print(tabulate(df.head(10), headers='keys', tablefmt='pretty'))

print(df)

columns_2 = ['강수량', '적설']

df_not_scaled = df.drop(columns=['강수량', '적설'])

# Min-max Scaler 이용
min_max_scaler = MinMaxScaler()

df_minmax = np.log1p(df[columns_2])
df_minmax = min_max_scaler.fit_transform(df_minmax)
df_minmax = pd.DataFrame(df_minmax, columns=columns_2)

df = pd.concat([df_not_scaled, df_minmax], axis=1)

print(tabulate(df[df['사용일자'] == 20231026], headers='keys', tablefmt='pretty'))

print(df)
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

def one_hot_encoding(origin_df, attribute):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = one_hot_encoder.fit_transform(origin_df[[attribute]])

    # 인코딩된 열 이름 얻기
    encoded_column_names = one_hot_encoder.get_feature_names_out([attribute])

    # 인코딩된 열을 데이터프레임으로 변환
    encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names)

    # 원래 데이터프레임과 인코딩된 열을 합치기
    origin_df = pd.concat([origin_df, encoded_df], axis=1)

    # 원래 열 삭제
    origin_df.drop(attribute, axis=1, inplace=True)

    return origin_df


# One-Hot Encoding
df = one_hot_encoding(df, "행정구역")

df.to_csv('merged_data/data_after_scaling.csv')

# 히스토그램과 KDE 플롯
def plot_distributions(df, columns, title):
    plt.figure(figsize=(14, 7))
    for col in columns:
        sns.histplot(df[col], kde=True, label=col, bins=30)
    plt.legend()
    plt.title(title)
    plt.show()

# 강수량, 적설량, 총이용승객수의 분포 시각화
#plot_distributions(df_scaled, ['강수량', '적설', '총이용승객수'], '강수량, 적설량, 총이용승객수 분포')

# 특정 컬럼의 분포 시각화 예시 (강수량)
plt.figure(figsize=(12, 6))
sns.histplot(df['강수량'], kde=True)
plt.title('강수량 분포')
plt.show()

# 특정 컬럼의 분포 시각화 예시 (적설량)
plt.figure(figsize=(12, 6))
sns.histplot(df['적설'], kde=True)
plt.title('적설량 분포')
plt.show()

# 특정 컬럼의 분포 시각화 예시 (총이용승객수)
plt.figure(figsize=(12, 6))
sns.histplot(df['평균이용승객수'], kde=True)
plt.title('평균이용승객수 분포')
plt.show()
