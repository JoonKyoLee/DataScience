import pandas as pd
import numpy as np
import openpyxl

station_origin_info = pd.read_excel('origin_data/도시철도역사정보.xlsx')
gyeongi_info = pd.read_csv('origin_data/경기_행정구역.csv')


def modify_station_info(attribute, existing_value, new_value):
    station_origin_info[attribute] = station_origin_info[attribute].str.replace(existing_value, new_value, regex=True)


# '역사명' 형식을 통일하기 위해서 마지막에 붙은 '역'을 제거
modify_station_info('역사명', '역$', '')
#station_origin_info['역사명'] = station_origin_info['역사명'].str.replace('역$', '', regex=True)

# '역사도로명주소' 속성에서 '시/도'와 '구/시' 추출
station_origin_info['시/도'] = station_origin_info['역사도로명주소'].apply(lambda x: x.split()[0] if pd.notnull(x) else '')
station_origin_info['구/시'] = station_origin_info['역사도로명주소'].apply(lambda x: x.split()[1] if pd.notnull(x) else '')

# '시/도'에서 마지막에 존재하는 시/도를 제거 (서울특별시 -> 서울, 서울시 -> 서울, 경기도 -> 경기)
modify_station_info('시/도', '특별시$', '')
#station_origin_info['시/도'] = station_origin_info['시/도'].str.replace('특별시$', '', regex=True)
station_origin_info['시/도'] = station_origin_info['시/도'].str.replace('광역시$', '', regex=True)
station_origin_info['시/도'] = station_origin_info['시/도'].str.replace('시$', '', regex=True)
station_origin_info['시/도'] = station_origin_info['시/도'].str.replace('도$', '', regex=True)

# '역사명'에서 괄호와 그 안의 내용을 제거
station_origin_info['역사명'] = station_origin_info['역사명'].str.replace(r'\(.*\)', '', regex=True)

# 역사명, 시/도, 구/시 속성만 새로운 데이터 프레임에 저장
station_info = station_origin_info[['역사명', '시/도', '구/시']]

# 서울, 경기, 인천 이외 지역은 데이터 프레임에서 drop
target_city = ['서울', '경기', '인천']
station_info = station_info[station_info['시/도'].isin(target_city)]

# '행정구역' 속성 추가
station_info['행정구역'] = station_info['시/도']

# 서울과 인천은 행정구역에 그대로 반영
station_info.loc[station_info['시/도'] == '서울', '행정구역'] = '서울'
station_info.loc[station_info['시/도'] == '인천', '행정구역'] = '인천'


# 경기도 행정구역 매핑 함수 정의
def get_gyeonggi_district(row):
    if row['시/도'] == '경기':
        match = gyeongi_info[gyeongi_info['시/군'] == row['구/시']]
        if not match.empty:
            return match.iloc[0]['행정구역']
    return row['행정구역']


# 경기도에 해당하는 행정구역 값 채우기
station_info.loc[station_info['시/도'] == '경기', '행정구역'] = station_info[station_info['시/도'] == '경기'].apply(get_gyeonggi_district, axis=1)

station_administrative_division = station_info[['역사명', '행정구역']]

# csv 파일로 저장
station_administrative_division.to_excel('modified_data/추출된_역정보.xlsx', index=False)

print(station_info['행정구역'].unique())
