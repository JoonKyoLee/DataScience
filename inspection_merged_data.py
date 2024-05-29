import pandas as pd
from tabulate import tabulate

merged_data = pd.read_csv('merged_data/modified_merged_data.csv')

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
