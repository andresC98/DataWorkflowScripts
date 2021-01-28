# Python applied to Data Science code snippets.
# Libraries: Pandas, Numpy, Matplotlib, Scikit-learn
# Author: Andrés Carrillo (github.com/AndresC98)


##############
### PANDAS ###
##############
import pandas as pd

## Defining a Pandas series (one-dimensional data structure)
my_array = [1, 3, 5, 6, 7, 8]
series = pd.Series(data=my_array)
## Computing std and mean of a Series
series.std(axis=None, skipna=None)
series.mean()
series.median()
## Defining a Pandas DataFrame 
my_2darr = [[1,2,3], [4,5,6]]
my_df = pd.DataFrame(data=my_2darr, columns=["A", "B","C"])
# results in: A0: 1, A1: 4, B0:2, B1: 5,...
# If we compute my_df.mean() will compute mean per column. (A=2.5, B=3.5,...)
## Chaning the indexes of dataframe:
my_df.reindex(["X", "Y"]) # AX: 1, AY: 4, BX: ....

# We can also create dataframe from a dictionary automatically:
my_dict = {"ID": [1, 4, 3], "Dept.": ["Aero", "TSC", "Bio"]}
my_df = pd.DataFrame(my_dict)
"""    ID      Dept
0      1        Aero
1      4        TSC
2      3        Bio """

# Creating a copy of a df or series:
copied_series = series.copy(deep=True)

# Adding a column to existing dataframe
my_df["NewCol"] = [4.4, 2.3, 5.0]
# Removing (dropping) column from datafrmae:
my_df.drop(labels=["ColA"], axis=1, inplace=True)
# Removing (dropping) row n from dataframe
my_df.drop(n,axis=0, inplace=True)
# Removing duplicate rows from DataFrame columns
my_df.drop_duplicates(["ColA", "ColB"])

# Retrieve list of indexes of a dataframe:
indexes = list(my_df.index)
# Retrieve list of columns of a dataframe:
cols = list(my_df.columns)
# Retrieve list of values of a dataframe column:
vals = my_df.values
# Get types of each column:
my_df.dtypes
# Get nº of elements (rows x cols) of df:
my_df.size
# Get shape (rows, cols) of a dataframe:
my_df.shape
# Get first n rows of dataframe
my_df.head(n) # .tail(n) to get last n

# access data in dataframe by index:
my_df.loc[0] # returns [A0, B0, C0]
# Note: if index were letters, also useful.
# Indexing by index position
my_df.iloc[0] # retirms [A0, B0, C0] (first index

## Boolean indexing: The condition inside bracket returns TRUE/FALSE matrix
# Series where value is greater than a given number
my_series[my_series > 5] # or NOT greater: my_series[~(my_series > 5)]
# Series within a range
my_series[(my_series < -4) | (my_series > 3)]
# Filtering dataframe
my_df[my_df['Population'] > 870000]

# Iterating over dataframe rows as (index, Series) pairs
for index, row in my_df.iterrows():
    print(row["A"], row["B"])

# Getting correlation matrix of a dataframe
my_df.corr()
# Getting counts of non-nulls (not NaNs) elements of dataframe columns:
my_df.count()
# Getting frequency counts of unique items 
my_df.value_counts()
# Getting number of unique values per column
my_df.nunique()
# Getting minimum or maximum values
my_df.min() # my_df.max()
# Obtain summary statistics of a dataframe per column
my_df.describe()

# Selecting subset of columns from dataframe:
my_df[["A", "C"]]
# Dropping NaNs of a dataframe:
clean_df = my_df.dropna()

# Checking if is NaN (returns boolean matrix)
my_df.isna()
# Replacing nan values in a dataframe column:
my_df["A"].replace(np.nan, my_df["A"].mean(), inplace=True) # Example imputing by mean
# Replacing certain values by NaN
my_df["A"].replace(["Unknown", "-", "?"], np.nan, inplace=True)
# Sorting dataframe by values of a certain column
my_df.sort_values(by=["A"])

# Applying function (example: sqrt) to specific column values
my_df["A"] = my_df["A"].apply(np.sqrt)
# Applying dynamic function (lambda function)
my_df["A"] = my_df["A"].apply(lambda x: x + 1)
# Another example but with map: (better for single cols)
my_df["A"] = my_df["A"].map(lambda a: a / 2.)

# Group by dataframe and apply aggregated function
grouped_df = my_df.groupby(by="Dept.").nunique()

# Append VS Concatenating dataframes
pd.concat([df1, df2])
my_df.join(df2, on="A")

# Check if values of columns are in certain range or list:
my_df.isin({"Dept.": ["Bio", "Maths", "CS"]})

# Compute min, 25th percentile, median, 75th perc. and max of a series
np.percentile(series, q=[0, 25, 50, 75, 100])

# Convert String col to datetime
from datetime import datetime
my_df["date"] = datetime.strptime(my_df["date"],"%m-%d-$Y")

# Saving dataframe to csv:
my_df.to_csv("mycsvdataframe.csv")
# Reading dataframe from csv:
new_df = pd.read_csv("data.csv", delimiter=";")
# Dataframe to Numpy array
my_df.to_numpy(dtype=float)

#################
###   Numpy   ###
#################
import numpy as np

# Creating numpy array:
arr = np.array([1,3,4,5])

# Get shape (dimensions) of an array or matrix
arr.shape

# Subsetting 2D numpy arrays : arr[rows, cols]
arr_2dims[:, 0] # will return all the rows from column 0
arr_2dims[:4, :1] # will return rows from 0 to 4, with columns 0 to 1.

# Remove item in position [i] from array
np.delete(arr, [1])

# Sorting numpy array (descending achieved with ::-1)
np.sort(arr)[::-1]

# Calculating percentiles:
np.percentile(arr, q=[0, 25, 50, 75, 100])

# Obtaining correlation coefficients:
arr.corrcoef()

# Obtain mean, median, std,...
np.mean(arr)
np.std(arr)
np.median(arr) 

# Generating evenly spaced numbers over interval
np.linspace(start=0, stop=20, num=50)

# Generating 10 times repeated element: 5
np.repeat(5, 10)

# Generating random numbers:
np.random.randint(low=1, high=100)
# Generating (n, m) matrix of random numbers:
np.random.randint(low=1, high=10, size=(n,m))
# Random choice from a list:
np.random.choice([1,2,3,4,5,6]) # dice roll
# Generate list of elements from random choice:
np.random.choice([1,2,3,4,5,6], size=10)

# Generating NaN value:
np.nan

# Get indices of maximum values along an axis:
max_pos = np.argmax(arr)
# Same but specific axis: Example: max value per column
max_pos_1 = np.argmax(arr, axis=1)

# Removing single dimensional entries from shape of array:
squeezed_arr = np.squeeze(arr, axis=0)

# Compute histogram from a set of data:
count, bid_edges = np.histogram(my_df["A"])

# Flattening a matrix
matr = np.array([1,2], [3,4])
flattened_matr = matr.flatten() # 1 2 3 4

# Vectorizing a function
def my_function(a, b, sc):
    return a*b + sc
vect_func = np.vectorize(my_function)
vect_funct(np.array([1,2]), np.array([4,3]), 15)

# Selecting from a generated list, a sample of 20 elements rounded
import random
sampls = np.round(random.sample(list(np.linspace(50, 200, 50)), 20))

####################
#### Matplotlib ####
####################

import matplotlib.pyplot as plt

# Plotting horizontal axis x vs transformation of x on y axis, on a figure
x = np.array([1,2,3,4,5,6,7,8])
y = pow(x, 2)
plt.figure()
plt.plot(x, y)
plt.show()

#... TODO: Add more

####################
### Scikit-learn ###
####################
import sklearn

# Vectorizing text in Tf-Idf (term freq. - inverse doc. freq.)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['First document is this.',
          'Here is the second document',
          'The first document is this?']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
X.toarray()


## Full example: loading dataset, train-test split, preprocessing, fitting model, obtaining acc.
from sklearn import datasets, preprocessing
from sklearn.neighbors import KNeighborsClassifier,
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading and splitting dataset into train-test partitions
iris_data = datasets.load_iris()
X, y = iris_data.data[:, :2], iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Standard scaling attributes
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Creating and fitting a model
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)
# Obtaining predictions and evaluating model
y_pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))
##


# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Imputing missing variables
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy='mean', axis=0) # mean imputation
X_train_imp = imputer.fit_transform(X_train)

# Obtaining regression metrics from regr. output. (MAE, MSE, RMSE...)
from sklearn.metrics import mean_absolute_error, mean_squared_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)) # OR mean_squared_error

# Performing cross-validation score over a model (model = knn, ...)
print(cross_val_score(model, X_train, y_train, cv=5))

# Model hyperparameter tuning via grid search
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,9), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn, param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

# Model hyperparam. tuning via random search
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,9), "weights": ["uniform", "distance"]}
random_search = RandomizedSearchCV(estimator=knn,
                                   param_distributions=params,
                                   cv=5,
                                   n_iter=8,
                                   random_state=42)
random_search.fit(X_train, y_train)
print(rsearch.best_score_)