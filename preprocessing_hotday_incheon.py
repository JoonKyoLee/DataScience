import pandas as pd

# 폭염일수 데이터 읽기
file_path = '/origin_data/2023_인천(강화)_폭염일수.csv'
hot_days = pd.read_csv(file_path, encoding='CP949')

# '지점명'을 '행정구역'으로 바꾸고 모든 값을 '인천'으로 통일
hot_days['행정구역'] = '인천'

# '지점번호' 및 '지점명' 열 제거
hot_days = hot_days.drop(columns=['지점번호', '지점명'])

# 열 이름 확인
print("변경된 열 이름:")
print(hot_days.columns)

# '날짜' 열을 datetime 형식으로 변환하고, 달 추출
hot_days['일자'] = pd.to_datetime(hot_days['일자'])
hot_days['달'] = hot_days['일자'].dt.to_period('M')

# 각 달의 최고기온 평균 계산
monthly_avg = hot_days.groupby(['행정구역', '달'])['최고기온'].mean().reset_index()

# '폭염일수' 열을 'monthly_avg' 데이터프레임에 추가
monthly_avg = monthly_avg.merge(monthly_heatwave_days, on=['행정구역', '달'])

# 폭염여부 추가
monthly_avg['폭염여부'] = monthly_avg['최고기온'].apply(lambda x: 'O' if x > 33 else 'X')

# 폭염 일수 계산
hot_days['폭염일수'] = hot_days['최고기온'].apply(lambda x: 1 if x > 33 else 0)
monthly_heatwave_days = hot_days.groupby(['행정구역', '달'])['폭염일수'].sum().reset_index()

# 폭염여부 추가
monthly_avg['폭염여부'] = monthly_avg['최고기온'].apply(lambda x: 'O' if x > 33 else 'X')


# 변경된 데이터 저장
output_path = '/modified_data/2023_인천(강화)_폭염일수_modified.csv'
monthly_avg.to_csv(output_path, index=False, encoding='CP949')

print(f"파일이 저장되었습니다: {output_path}")
