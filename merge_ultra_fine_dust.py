import pandas as pd

seoul = pd.read_csv('modified_data/수정된_초미세먼지_정보(서울).csv')
gyeongi = pd.read_csv('modified_data/수정된_초미세먼지_정보(경기).csv')
incheon = pd.read_csv('modified_data/수정된_초미세먼지_정보(인천).csv')

merged_ultra_fine_dust = pd.concat([seoul, gyeongi], ignore_index=True)
merged_ultra_fine_dust = pd.concat([merged_ultra_fine_dust, incheon], ignore_index=True)
print(merged_ultra_fine_dust)

merged_ultra_fine_dust.to_csv('modified_data/수정된_초미세먼지_정보.csv', index=False)
