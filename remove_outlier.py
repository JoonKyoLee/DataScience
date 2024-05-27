import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def saveDataset(df, file_path, file_name):
    df.to_csv(file_path + file_name + ".csv", encoding="utf-8")

def remove_outliers(df, column):
    # 이상치 제거를 위한 IQR 계산
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    # 이상치 범위 계산
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 이상치 제거
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def removeOutliers(data_list, data_name_list, columns, file_path):
    data_after_list = []
    for df, df_name in zip(data_list, data_name_list):
        for column in columns:
            df = remove_outliers(df, column)
            saveDataset(df, file_path, df_name)
        data_after_list.append(df)
    return data_after_list

def visualize_outliers(data_before, data_after, column):
    plt.figure(figsize=(12, 6))
    
    # 이상치 제거 전 데이터 시각화
    plt.subplot(1, 2, 1)
    plt.title(f"이상치 제거 전 ({column})")
    plt.boxplot(data_before[column])
    plt.ylabel("값")
    
    # 이상치 제거 후 데이터 시각화
    plt.subplot(1, 2, 2)
    plt.title(f"이상치 제거 후 ({column})")
    plt.boxplot(data_after[column])
    plt.ylabel("값")
    
    plt.tight_layout()
    plt.show()

def compare_outliers(data_list_before, data_list_after, column_list):
    for data_before, data_after, column in zip(data_list_before, data_list_after, column_list):
        visualize_outliers(data_before, data_after, column)

if __name__ == "__main__":
    # 데이터 불러오기
    file_path = "/merged_data/"  # 파일 경로 설정
    file_names = ["modified_merged_data.csv"]  # 파일 이름 설정
    data_list = [pd.read_csv(file_path + file_name) for file_name in file_names]

    # 승차 총 승객수와 하차 총 승객수를 합한 총 승객수 열 생성
    for df in data_list:
        df['총승객수'] = df['승차총승객수'] + df['하차총승객수']

    # 이상치 제거할 열 선택
    columns = ['총승객수']  # 총 승객수만 선택

    # 이상치 제거 및 결과 저장
    data_after_list = removeOutliers(data_list, file_names, columns, file_path)

    # 이상치 제거된 데이터 비교 시각화
    compare_outliers(data_list, data_after_list, columns)

    # 이상치 제거 전 데이터셋의 총 승객수에 대한 통계적 요약 출력
    print("\n이상치가 제거되기 전 데이터셋의 총 승객수에 대한 통계적 요약:")
    print(data_list[0]['총승객수'].describe())

    # 이상치 제거된 데이터 출력 및 기술 통계량 계산
    for idx, df_after in enumerate(data_after_list):
        print(f"\n이상치가 제거된 데이터 {file_names[idx]}:")
        print(df_after)
        print("\nRemove Outliers 기술 통계량:")
        print(df_after[columns].describe())
