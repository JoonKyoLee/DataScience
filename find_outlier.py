import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

data = pd.read_csv('merged_data/modified_merged_data.csv')
print(data)

data_for_remove_outlier = data

# .loc를 사용하여 명시적으로 값을 설정
data_for_remove_outlier['총승객수'] = data_for_remove_outlier['승차총승객수'] + data_for_remove_outlier['하차총승객수']

data_for_remove_outlier = data_for_remove_outlier[['사용일자', '총승객수']]

# '사용일자'별로 '총승객수'를 그룹화하여 합계 계산
grouped_df = data_for_remove_outlier.groupby('사용일자')['총승객수'].sum().reset_index()

# 결과 출력
print(grouped_df['총승객수'].mean())

print(grouped_df[grouped_df['사용일자'] == 20240328])

print('\n[일자별 총승객수 통계]')
stats = grouped_df['총승객수'].describe()
# 지수 표기법 대신 일반 표기법으로 출력하도록 설정 변경
pd.set_option('display.float_format', '{:.2f}'.format)
print(stats)
