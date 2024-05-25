import pandas as pd

# 폭염일수 데이터 읽기
file_path = '/origin_data/2023_인천(강화)_폭염일수.csv'
hot_days = pd.read_csv(file_path, encoding='CP949')

# '지점명'을 '행정구역'으로 바꾸고 모든 값을 '인천'으로 통일
hot_days['행정구역'] = '인천'

# '지점번호' 열 및 '지점명' 열 제거
hot_days = hot_days.drop(columns=['지점번호', '지점명'])

# '일자' 열을 'YYYYMMDD' 형식으로 변환
hot_days['일자'] = pd.to_datetime(hot_days['일자'], format='%Y.%m.%d').dt.strftime('%Y%m%d')

# '최고기온' 열 삭제
hot_days = hot_days.drop(columns=['최고기온'])

# 변경된 데이터 저장
output_path = '/modified_data/2023_인천(강화)_폭염일수_modified.csv'
hot_days.to_csv(output_path, index=False, encoding='CP949')

print(f"파일이 저장되었습니다: {output_path}")
