import pandas as pd

seoul = pd.read_csv('../modified_data/수정된_미세먼지_정보(서울).csv')
gyeongi = pd.read_csv('../modified_data/수정된_미세먼지_정보(경기).csv')
incheon = pd.read_csv('../modified_data/수정된_미세먼지_정보(인천).csv')

merged_fine_dust = pd.concat([seoul, gyeongi], ignore_index=True)
merged_fine_dust = pd.concat([merged_fine_dust, incheon], ignore_index=True)
print(merged_fine_dust)

# 경보단계 우선순위를 위한 사전 정의
priority = {'주의보': 1, '경보': 2}

# 경보단계에 우선순위 추가
merged_fine_dust['priority'] = merged_fine_dust['경보단계'].map(priority)

# '일자'와 '권역'을 기준으로 그룹화하여 우선순위가 높은 행을 선택
merged_fine_dust = merged_fine_dust.sort_values('priority', ascending=False).drop_duplicates(subset=['일자', '권역'])

# 'priority' 컬럼 삭제
merged_fine_dust = merged_fine_dust.drop(columns=['priority'])

# 결과 출력
print(merged_fine_dust)

merged_fine_dust.to_csv('modified_data/수정된_미세먼지_정보.csv', index=False)
