import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

#Loading in and cleaning data
honey = "../2_Transform/hp_prod_19.csv"
df1 = pd.read_csv(honey)
df1.head()
df1 = df1[df1['state']!='United States']
# df1 = df1[df1['state']!='Other States']
df1.head()
X = df1[['max_h_prod_cny','prod_held_stocks']]
y = df1['yield/cny'].astype(int)
feature_names = X
print(X.shape, y.shape)

#Making model
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
regr = make_pipeline(StandardScaler(),LinearSVR(random_state=23, tol=1e-5))
regr.fit(X, y)
print(regr.predict([[input(),input()]]))
