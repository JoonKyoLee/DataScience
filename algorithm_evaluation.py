import pandas as pd
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import seaborn as sns

# 한글 폰트 설정 (예: 나눔고딕)
font_path = '../Library/Fonts/KoPubWorld Dotum Medium.ttf'
fontprop = fm.FontProperties(fname=font_path, size=10)
plt.rc('font', family=fontprop.get_name())

# Load and preprocess data
df = pd.read_csv('merged_data/final_data.csv')

df.drop(columns=['Unnamed: 0', '사용일자'], inplace=True)

def select_model(task, model_type):
    if task == 'classification':
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Model Type is wrong.")
    elif task == 'regression':
        if model_type == 'linear_regression':
            return LinearRegression()
        elif model_type == 'random_forest':
            return RandomForestRegressor(random_state=42)
        elif model_type == 'decision_tree':
            return DecisionTreeRegressor(random_state=42)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError("Model Type is wrong.")
    else:
        raise ValueError("You have to select classification or regression.")

def evaluate_model(model, X_train, X_test, y_train, y_test, task, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == 'classification':
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for {model_name} (Classification)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    elif task == 'regression':
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 Score: {r2_score(y_test, y_pred)}")

        plt.figure(figsize=(10, 7))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Regression Results for {model_name} (Regression)")
        plt.show()

    return model

def plot_feature_importance(model, features, task):
    if task == 'classification' and hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif task == 'regression' and hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")

    features_names = features.columns

    feature_importance_df = pd.DataFrame({'Feature': features_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance in {model.__class__.__name__} ({task.capitalize()})')
    plt.show()

def main(task, model_type):
    # Define features and target
    if task == 'classification':
        features = df.drop(columns=['평균이용승객수'])
        target = (df['평균이용승객수'] > df['평균이용승객수'].median()).astype(int)
    elif task == 'regression':
        target = df['평균이용승객수']
        features = df.drop(columns=['평균이용승객수'])
    else:
        raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select model
    model = select_model(task, model_type)

    # Evaluate model
    model = evaluate_model(model, X_train, X_test, y_train, y_test, task, model.__class__.__name__)

    # Plot feature importance if applicable
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor)):
        plot_feature_importance(model, features, task)


main('classification', 'random_forest')
main('regression', 'random_forest')
main('regression', 'linear_regression')
main('regression', 'decision_tree')
main('regression', 'gradient_boosting')
