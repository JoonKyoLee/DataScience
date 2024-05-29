import pandas as pd

# 데이터 읽기
weather = pd.read_csv('../origin_data/서울경기인천_일별_시간대별_날씨정보_기온_강수량.csv', encoding='EUC-KR')
city = pd.read_csv('../origin_data/강수량_적설_행정구역.csv')

# '지점명', '일시', '강수량(mm)', '적설(cm)' 열만 선택
weather = weather[['지점명', '일시', '강수량(mm)', '적설(cm)']]

weather = pd.merge(weather, city, on='지점명', how='left')

# '일시' 열을 '날짜'와 '시간' 열로 분리
weather[['날짜', '시간']] = weather['일시'].str.split(' ', expand=True)

# 강수량(mm)와 적설(cm) 열의 NaN 값을 0으로 대체
weather['강수량(mm)'].fillna(0, inplace=True)
weather['적설(cm)'].fillna(0, inplace=True)

# '1:00', '2:00', '3:00', '4:00', '5:00' 시간을 제외하고 필터링
weather_filtered = weather[~weather['시간'].isin(['1:00', '2:00', '3:00', '4:00', '5:00'])]
#print(weather_filtered.head(20))

# '지점명'과 '날짜'를 기준으로 그룹화하여 강수량과 적설의 평균 계산
weather_grouped = weather_filtered.groupby(['행정구역', '날짜']).agg({'강수량(mm)': 'mean', '적설(cm)': 'mean'}).reset_index()

# 날짜 형식을 datetime으로 변환 후 'YYYYMMDD' 형식으로 변환
weather_grouped['날짜'] = pd.to_datetime(weather_grouped['날짜'], format='%Y.%m.%d').dt.strftime('%Y%m%d')

weather_grouped = weather_grouped.sort_values(by='날짜')

weather_grouped.to_csv('modified_data/수정된_강수량_적설_정보.csv', index=False)
