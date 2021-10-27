import pandas as pd
from sklearn import metrics


def fit_predict(classifiers, x_train, y_train, x_test, y_test):
    results = []
    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        r2_score = metrics.r2_score(prediction, y_test)
        results.append({
            'ModelName': name,
            'R2 Score': r2_score
        })

    results = pd.DataFrame(results)
    return results
