import pandas as pd

passenger_origin = pd.read_csv('../origin_data/승하차_인원정보.csv')

# 역명이 최근에 수정된 뚝섬유원지 역명을 자양(뚝섬한강공원)으로 수정
passenger_origin['역명'] = passenger_origin['역명'].str.replace('뚝섬유원지', '자양(뚝섬한강공원)')

# '역사명'에서 괄호와 그 안의 내용을 제거
passenger_origin['역명'] = passenger_origin['역명'].str.replace(r'\(.*\)', '', regex=True)

station_info = pd.read_excel('../modified_data/추출된_역정보.xlsx')

merged_passenger = pd.merge(passenger_origin, station_info, left_on='역명', right_on='역사명', how='left')

# 등록일자를 제외 후 데이터 프레임 생성
modified_passenger = merged_passenger.drop(columns=['등록일자', '역사명'])

# 행정구역이 null인 데이터 삭제
modified_passenger = modified_passenger.dropna(subset=['행정구역'])

print(modified_passenger)

# 엑셀 파일로 저장
modified_passenger.to_excel('modified_data/수정된_승하차_인원정보.xlsx', index=False)
