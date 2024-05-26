import pandas as pd
from tabulate import tabulate

merged_data = pd.read_csv('merged_data/merged_data.csv')

print("[Features]")
print(merged_data.columns)

stats = merged_data.describe()
print()
print(tabulate(stats, headers='keys', tablefmt='pretty'))

print("\n[Data types]")
print(merged_data.dtypes)

print()
print(merged_data.head(5))

# 각 속성의 NaN 값의 개수 출력
nan_counts = merged_data.isna().sum()
print("\n[Number of NaN values]")
print(nan_counts)

# 행(row)과 열(column) 개수 출력
rows, columns = merged_data.shape
print()
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")


# NaN 처리
# 강수량과 적설의 NaN 값을 0으로 대체
merged_data['강수량'] = merged_data['강수량'].fillna(0)
merged_data['적설'] = merged_data['적설'].fillna(0)

# 나머지 열의 NaN 값을 'X'로 대체
columns_to_fill_with_X = ['미세먼지', '초미세먼지', '폭염여부', '한파특보', '황사관측']

for column in columns_to_fill_with_X:
    merged_data[column] = merged_data[column].fillna('X')

# 각 속성의 NaN 값의 개수 출력
nan_counts = merged_data.isna().sum()
print("\n[Number of NaN values]")
print(nan_counts)


# NaN 처리한 데이터를 csv로 저장
merged_data.to_csv("merged_data/modified_merged_data.csv", index=False)
