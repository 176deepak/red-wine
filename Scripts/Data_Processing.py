#required modules
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle


#function for handling imbalace dataset by oversampling method using RandomOverSampler of imblearn module 
def data_balancer(X, y):
    try:
        over_sampler = RandomOverSampler(random_state = 42)
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        return (X_resampled, y_resampled)
    except:
        print("Something else went wrong in data_balancer")


#function for selction of features which is correlated mostly
def feature_selector(dataset, threshold):
    try:
        col_corr = set()  
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]  
                    col_corr.add(colname)
        return col_corr
    except:
        print("Something else went wrong in feature_selector")


# function for scaling dataset columns into same scale or -1 to 1 
def data_transformer(X, y):
    try:
        print("Step 1:")
        numeric_features = list(X.columns)
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(
        transformers=[('numeric', numeric_transformer, numeric_features)])
        pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor)  
        ])
        print("step 2")
        X = pipeline.fit_transform(X)
        print("step 3")
        with open('Models\Saved_Models\preprocessor.pkl', 'wb') as file:
            pickle.dump(pipeline, file)
        print("step 4")
        
        # cols = list(X.columns)
        # scaler = StandardScaler()
        # scaler.fit(X)
        # X = scaler.transform(X)
        print("step 5")    
        X = pd.DataFrame(X, columns=numeric_features)
        y = y.apply(lambda x: 1 if x<7 else 0)
        return (X, y)
    except:
        print("Something else went wrong")

#function for spliting datasets into train and test datasets
def train_test_data(X,y):
    try:
        X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return (X_train , X_test, y_train, y_test)
    except:
        print("Something else went wrong")


def data_dump(X, y, path):
    try:
        data = pd.concat([X, y], axis=1)
        print('concetenation done!')
        data.to_csv(path, index=False)
    except:
        print("Something else went wrong in data_dump")

#loading dataset from data_source folder
df = pd.read_csv("Data\Raw\winequality-red.csv")

#deleting duplicate values from dataset
df.drop_duplicates(inplace=True, ignore_index = True)

#converting dataset into independent and dependent dataset
X = df.drop(['quality'], axis=1)
y = df['quality']

#balancing imbalance data
X, y = data_balancer(X, y)

#extra columns which is not have much information
corr_features = feature_selector(X, 0.7)
X = X.drop(['citric acid'], axis=1)
data_dump(X, y, 'Data\Features\engineered_features.csv')

#scaling data into -1 to 1
X, y = data_transformer(X, y) 
data_dump(X, y, 'Data\Processed\preprocessed_data.csv')

#spliting data into train and test datasets
X_train, X_test, y_train, y_test = train_test_data(X, y)
data_dump(X_train, y_train, r'Data\Splits\train.csv')
data_dump(X_test, y_test, r'Data\Splits\test.csv')