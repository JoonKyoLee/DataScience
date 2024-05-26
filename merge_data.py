import pandas as pd

subway_passenger = pd.read_excel('modified_data/수정된_승하차_인원정보.xlsx')


# 미세먼지 데이터 병합
fine_dust = pd.read_csv('modified_data/수정된_미세먼지_정보.csv')

merged_df = pd.merge(subway_passenger, fine_dust, left_on=['사용일자', '행정구역'], right_on=['일자', '권역'], how='left')

# 미세먼지 데이터에 존재하는 경보단계를 미세먼지로 변경
merged_df.rename(columns={'경보단계': '미세먼지'}, inplace=True)

# merge 후에 불필요한 열 제거
merged_df.drop(columns=['물질명', '권역', '일자'], inplace=True)


# 초미세먼지 데이터 병합
ultra_fine_dust = pd.read_csv('modified_data/수정된_미세먼지_정보.csv')

merged_df = pd.merge(merged_df, ultra_fine_dust, left_on=['사용일자', '행정구역'], right_on=['일자', '권역'], how='left')

# 초미세먼지 데이터에 존재하는 경보단계를 미세먼지로 변경
merged_df.rename(columns={'경보단계': '초미세먼지'}, inplace=True)

# merge 후에 불필요한 열 제거
merged_df.drop(columns=['물질명', '권역', '일자'], inplace=True)


# 강수량, 적설량 데이터 병합
rain_snow = pd.read_csv('modified_data/수정된_강수량_적설_정보.csv')

merged_df = pd.merge(merged_df, rain_snow, left_on=['사용일자', '행정구역'], right_on=['날짜', '행정구역'], how='left')

# merge 후에 불필요한 열 제거
merged_df.drop(columns=['날짜'], inplace=True)


# 폭염 데이터 병합
hot = pd.read_csv('modified_data/수정된_폭염_정보.csv')

merged_df = pd.merge(merged_df, hot, left_on=['사용일자', '행정구역'], right_on=['일자', '행정구역'], how='left')

merged_df.drop(columns=['일자'], inplace=True)


# 한파 데이터 병합
cold = pd.read_csv('modified_data/수정된_한파_정보.csv')

merged_df = pd.merge(merged_df, cold, left_on=['사용일자', '행정구역'], right_on=['일시', '행정구역'], how='left')

merged_df.drop(columns=['일시'], inplace=True)


# 황사 데이터 병합
dust = pd.read_csv('modified_data/수정된_황사_정보.csv')


# '경기' 지역명 처리 함수 정의
def is_gyeonggi(region):
    if region.startswith('경기'):
        return '경기'
    return region


merged_df['행정구역_병합'] = merged_df['행정구역'].apply(is_gyeonggi)

merged_df = pd.merge(merged_df, dust, left_on=['행정구역_병합', '사용일자'], right_on=['지점명', '일자'], how='left')

merged_df.drop(columns=['행정구역_병합', '지점명', '일자'], inplace=True)


merged_df.rename(columns={'강수량(mm)': '강수량'}, inplace=True)
merged_df.rename(columns={'적설(cm)': '적설'}, inplace=True)
merged_df.rename(columns={'폭염여부(O/X)': '폭염여부'}, inplace=True)
merged_df.rename(columns={'한파특보(O/X)': '한파특보'}, inplace=True)
merged_df.rename(columns={'황사관측(O/X)': '황사관측'}, inplace=True)

print(merged_df)

merged_df.to_csv('merged_data/merged_data.csv', index=False)
