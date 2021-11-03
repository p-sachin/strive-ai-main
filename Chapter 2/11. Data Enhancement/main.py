# videos.py
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
import time
from sklearn import impute
from sklearn import compose
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_dataframe(path):
    df = pd.read_csv(path)
    # Splitting the timestamp column into 3 columns namely years, months and hours
    df['year'] = df['timestamp'].apply(lambda row: row[:4])
    df['month'] = df['timestamp'].apply(lambda row: row.split('-')[1])
    df['hour'] = df['timestamp'].apply(lambda row: row.split(':')[0][-2:])
    df.drop('timestamp', axis=1, inplace=True)
    return df


def data_enhancement(data, value):
    gen_data = data
    for season in data['season'].unique():
        seasonal_data = gen_data[gen_data['season'] == season]
        hum_std = seasonal_data['hum'].std()
        wind_speed_std = seasonal_data['wind_speed'].std()
        t1_std = seasonal_data['t1'].std()
        t2_std = seasonal_data['t2'].std()
        for i in gen_data[gen_data['season'] == season].index:
            if np.random.randint(2) == 1:
                gen_data['hum'].values[i] += hum_std / 10
            else:
                gen_data['hum'].values[i] -= hum_std / 10
            if np.random.randint(2) == 1:
                gen_data['wind_speed'].values[i] += wind_speed_std / 10
            else:
                gen_data['wind_speed'].values[i] -= wind_speed_std / 10
            if np.random.randint(2) == 1:
                gen_data['t1'].values[i] += t1_std / 10
            else:
                gen_data['t1'].values[i] -= t1_std / 10
            if np.random.randint(2) == 1:
                gen_data['t2'].values[i] += t2_std / 10
            else:
                gen_data['t2'].values[i] -= t2_std / 10

        samples = gen_data.sample(gen_data.shape[0] // value)
        return samples


def get_scores(model, train_x, train_y, test, predict):
    results = []
    for name, model in model.items():
        start_time = time.time()
        model.fit(train_x, train_y.ravel())
        total_time = time.time() - start_time
        prediction = model.predict(test)
        results.append({
            'ModelName': name,
            'MSE': metrics.mean_squared_error(prediction, predict),
            'MAE': metrics.mean_absolute_error(prediction, predict),
            'Time': total_time})
    return pd.DataFrame(results)



# Creating a new dataframe df_bike
df_bike = get_dataframe("./data/london_merged.csv")


# Splitting features and the target variables from the dataset
X = df_bike.iloc[:,1:]
y = df_bike.iloc[:,0]

# Data Preprocessing
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
# Adding new training samples (25%) from the enhanced dataset
extra_sample = data_enhancement(df_bike, 4)
x_train = pd.concat([x_train, extra_sample.drop(['cnt'], axis=1)])
y_train = pd.concat([y_train, extra_sample['cnt']])
transformer = preprocessing.MinMaxScaler()
y_train = transformer.fit_transform(y_train.values.reshape(-1, 1))
y_test = transformer.transform(y_test.values.reshape(-1, 1))

# Checking the correlation of each features in the given dataset
correlation = df_bike.corr()
corr_features = correlation.index
plt.figure(figsize=(10,8))
fig = sns.heatmap(df_bike[corr_features].corr(), annot=True, cmap="YlGnBu")

# Feature Selection using SelectKBest
X_new = SelectKBest(f_regression, k=6).fit_transform(x_train, y_train.ravel())

cat_vars = ['hour', 'weather_code']
num_vars = ['t1', 't2', 'hum', 'wind_speed']

num_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999)),
])
cat_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
    ('ordinal', preprocessing.OrdinalEncoder())  # handle_unknown='ignore' ONLY IN VERSION 0.24
])
tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    ('cat', cat_4_treeModels, cat_vars),
], remainder='drop')  # Drop other vars not specified in num_vars or cat_vars

tree_classifiers = {
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100),
}

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

print(get_scores(tree_classifiers, x_train, y_train, x_test, y_test))

