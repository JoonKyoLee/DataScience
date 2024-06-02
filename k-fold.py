import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Load and preprocess data
df = pd.read_csv('merged_data/final_data_with_weather.csv')
df.drop(columns=['Unnamed: 0', '사용일자'], inplace=True)


def select_model(model_type):
    if model_type == 'logistic_regression':
        return LogisticRegression(random_state=42)
    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        return SVC(random_state=42)
    else:
        raise ValueError("Unsupported model type for classification.")


def kfold_cross_validation(model_types, n_splits_range):
    features = df.drop(columns=['평균이용승객수'])
    target = (df['평균이용승객수'] > df['평균이용승객수'].median()).astype(int)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    kfold_results = {}

    for model_name in model_types:
        model = select_model(model_name)
        results = []

        for n_splits in n_splits_range:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, features, target, cv=kf, scoring='accuracy')
            results.append((n_splits, np.mean(scores)))

        kfold_results[model_name] = results

    plot_kfold_results(kfold_results)


def plot_kfold_results(kfold_results):
    plt.figure(figsize=(10, 7))
    for model_name, results in kfold_results.items():
        folds, scores = zip(*results)
        plt.plot(folds, scores, label=model_name, marker='o', linestyle='--')

    plt.xlabel('Number of folds')
    plt.ylabel('Cross-validation Score')
    plt.title('Cross-validation Score vs. Number of Folds')
    plt.legend()
    plt.show()


n_splits_range = range(2, 16)
model_types_classification = ['logistic_regression', 'random_forest', 'svm']
kfold_cross_validation(model_types_classification, n_splits_range)
