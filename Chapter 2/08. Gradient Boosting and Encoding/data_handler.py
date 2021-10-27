import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def get_data(pth):
    data = pd.read_csv(pth)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
    return x_train, x_test, y_train, y_test


def data_transform(x):
    categorical_features = x.select_dtypes(include=[object]).columns.values.tolist()
    categorical_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    #categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features)
    ], remainder='passthrough')
    data = preprocessor.fit_transform(x)
        
    return data

