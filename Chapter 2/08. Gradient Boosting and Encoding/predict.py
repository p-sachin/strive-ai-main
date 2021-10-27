import data_handler as dh
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

x_train, x_test, y_train, y_test = dh.get_data("insurance.csv")


def convert_to_dt(data):
    column = ["age", "sex", "bmi", "children", "smoke", "region"]
    test_data = pd.DataFrame([data], columns=column)
    return test_data


model = GradientBoostingRegressor()

check = True
while check:
    age = int(input("How old are you? \n"))
    sex = str(input("Which gender are you? \n"))
    bmi = float(input("What is your BMI? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    region = str(input("Which region are you from? \n"))
    check_data = [age, sex, bmi, child, smoke, region]
    check_data = convert_to_dt(check_data)
    check_data = dh.data_transform(check_data)
    model.fit(dh.data_transform(x_train), y_train)
    prediction = model.predict(check_data)
    print("The estimated insurance cost is:", prediction[0])
    check = False
