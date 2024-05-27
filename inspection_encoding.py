import pandas as pd
from tabulate import tabulate

df = pd.read_csv('merged_data/modified_merged_data.csv')

print(tabulate(df.head(10), headers='keys', tablefmt='pretty'))
