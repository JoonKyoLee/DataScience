import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


### detect dirty data - detect and print count

def detectDirtyData(df):
    
    line = pd.read_csv('merged_data/modified_merged_data.csv',nrows=1200)['노선명'].unique()
    line = np.hstack((line, '서해선'))  #230701 신설
    station = pd.read_excel('modified_data/추출된_역정보.xlsx')['역사명'].unique()
    district = pd.read_excel('modified_data/추출된_역정보.xlsx')['행정구역'].unique()
    count = 0
    
    for i, row in df.iterrows():
        
        # Out of date range
        if row['사용일자'] < 20230401 or row['사용일자'] > 20240331:
            count += 1
            continue
        
        # Wrong line name
        if row['노선명'] not in line:
            count += 1
            continue
        
        # Wrong station name
        if row['역명'] not in station:
            count += 1
            continue
        
        # Wrong district name
        if row['행정구역'] not in district:
            count += 1
            continue
        
        # Too few passengers
        if row['승차총승객수'] < 20 or row['하차총승객수'] < 20:
            count += 1
            continue
        
        # negative amount
        if row['강수량'] < 0 or row['적설'] < 0:
            count += 1
            continue
        
    print("number of dirty data:", count)
    print()



### add dirty data
# appendDirtyList() - create random number of dirty data and append to the list
# addDirtyData() - add dirty data by appendDirtyList() and create new csv file

def appendDirtyList(column_name, value, list):
    # github merge 후 바뀐 column대로 수정해야함
    ex_row = {'사용일자':20230715, '노선명':'분당선', '역명':'가천대', '승차총승객수':5256, '하차총승객수':5427, 
               '행정구역':'경기중부', '미세먼지':'X', '초미세먼지':'X', '강수량':0.1631578947368421, '적설':0.0, 
               '폭염여부':'X', '한파특보':'X', '황사관측':'X'}
    
    ex_row[column_name] = value
    n = random.randint(0, 100)
    for i in range(n):
        list.append(ex_row)
    
    return list

def addDirtyData():
    
    df = pd.read_csv('merged_data/modified_merged_data.csv')
    
    ex_row = {'사용일자':[20230715], '노선명':['분당선'], '역명':['가천대'], '승차총승객수':[5256], '하차총승객수':[5427], 
               '행정구역':['경기중부'], '미세먼지':['X'], '초미세먼지':['X'], '강수량':[0.1631578947368421], '적설':[0.0], 
               '폭염여부':['X'], '한파특보':['X'], '황사관측':['X']}
    dirty = []
    count = 0
    
    # Out of date range
    dirty = appendDirtyList('사용일자', 20220101, dirty)
        
    # 오류 처리 후 주석 해제 예정
    # # Wrong line name
    # dirty = appendDirtyList('노선명', '12호선', dirty)
        
    # # Wrong station name
    # dirty = appendDirtyList('역명', '가천대역', dirty)
        
    # # Wrong district name
    # dirty = appendDirtyList('행정구역', '부산', dirty)
        
    # Too few passengers
    dirty = appendDirtyList('승차총승객수', 2, dirty)
        
    # negative amount
    dirty = appendDirtyList('강수량', -1.23, dirty)

    dirty_df = pd.DataFrame(dirty)
    result_df = pd.concat([df, dirty_df])   # add dirty data
    result_df.to_csv("merged_data/data_after_add_dirtydata.csv", index=False)
    
    print('- Added dirty data -')
    print(dirty_df)
    print()
    
    
# test
# addDirtyData()



### remove dirty data - remove dirty data and save in new csv file

def removeDirtyData():

    df = pd.read_csv('merged_data/data_after_add_dirtydata.csv')
    
    line = pd.read_csv('merged_data/modified_merged_data.csv',nrows=1200)['노선명'].unique()
    line = np.hstack((line, '서해선'))  #230701 신설
    station = pd.read_excel('modified_data/추출된_역정보.xlsx')['역사명'].unique()
    district = pd.read_excel('modified_data/추출된_역정보.xlsx')['행정구역'].unique()
    
    # Out of date range
    df = df[df['사용일자'] >= 20230401]
    df = df[df['사용일자'] <= 20240331]
    # valid date 처리는 일단 제외
    
    # 오류 처리 후 주석 해제 예정
    # count = 0
    # for i, row in df.iterrows():
    #     # Wrong line name
    #     if row['노선명'] not in line:
    #         df.drop(index=[i])
    #         count += 1
    #         continue
        
    #     # Wrong station name
    #     if row['역명'] not in station:
    #         df.drop(index=[i])
    #         count += 1
    #         continue
        
    #     # Wrong district name
    #     if row['행정구역'] not in district:
    #         df.drop(index=[i])
    #         count += 1
    #         continue
    # print(count)    
        
    # Too few passengers
    df = df[df['승차총승객수'] >= 20]
    df = df[df['하차총승객수'] >= 20]
        
    # negative amount
    df = df[df['강수량'] >= 0]
    df = df[df['적설'] >= 0]

    df.to_csv("merged_data/data_after_remove_dirtydata.csv", index=False)    


# test
# removeDirtyData()


# final test
print('- Before adding dirty data -')
detectDirtyData(pd.read_csv('merged_data/modified_merged_data.csv')) 

addDirtyData()
print('- After adding dirty data -')
detectDirtyData(pd.read_csv('merged_data/data_after_add_dirtydata.csv')) 

removeDirtyData()
print('- After removing dirty data -')
detectDirtyData(pd.read_csv('merged_data/data_after_remove_dirtydata.csv'))


