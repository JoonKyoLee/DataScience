import pandas as pd

seoul = pd.read_csv('../modified_data/수정된_폭염_정보(서울).csv')
gyeongi = pd.read_csv('../modified_data/수정된_폭염_정보(경기).csv')
incheon = pd.read_csv('../modified_data/수정된_폭염_정보(인천).csv')

merge_hot = pd.concat([seoul, gyeongi], ignore_index=True)
merge_hot = pd.concat([merge_hot, incheon], ignore_index=True)
print(merge_hot)

merge_hot.to_csv('modified_data/수정된_폭염_정보.csv', index=False)
