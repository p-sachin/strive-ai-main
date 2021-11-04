from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

def data_preprocessing(path):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_preprocessing("heart.csv")

def column_transformer(x, y):
    cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
    num_cols = ["age","trtbps","chol","thalachh","oldpeak"]
    cols = ["age","trtbps","chol","thalachh","oldpeak", 'sex','exng','caa','cp','fbs','restecg','slp','thall']
    numerical_transformer = StandardScaler()
    ct = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_cols)
    ], remainder='passthrough')
    X_train_scaled = ct.fit_transform(x)
    X_test_scaled = ct.transform(y)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=cols)
    return X_train_scaled, X_test_scaled, ct

X_train_scaled, X_test_scaled, ct = column_transformer(X_train, X_test)

clf_tuned_trees = {
    'SVC': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'GradientBoosting': GradientBoostingClassifier(max_depth=1, n_estimators=250),
    'AdaBoostClassifier': AdaBoostClassifier()
}

def get_scores(x):
    results = []
    for name, model in x.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)
        test_score = model.score(X_test_scaled, y_test)
        accuracy = (y_test == prediction).mean()*100
        results.append({
        'ModelName': name,
        'Accuracy': accuracy,
        'Train Score': train_score,
        'Test Score': test_score
        })
    return pd.DataFrame(results)

tuned_scores_no_aug = get_scores(clf_tuned_trees)
print(tuned_scores_no_aug)

