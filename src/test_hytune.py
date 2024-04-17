"""
File by Sathiya Narayanan Venkatesan
This file contains the main functions of the project
"""


#importing the required packages
import os
import csv
import time
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def create_random_forest_regression_data_set(data_set):

    x_train, x_test, y_train, y_test = create_dataframe(data_set)

    #Generating data
    all_data = []

    for i in range(10000):
        start_time = time.time()
        # loop through and create 10000 random hyper sets
        n_estimators = random.randint(5, 800)
        max_depth = random.randint(0, 300)
        if max_depth == 0:
            max_depth = None
        min_samples_split = random.randint(2, 20)
        min_samples_leaf = random.randint(1, 50)
        max_features = random.sample([None, "sqrt", "log2"], 1)[0]
        bootstrap = random.sample([True, False], 1)[0]

        r_forest = RandomForestRegressor(random_state=0,
                                         n_estimators=n_estimators,
                                         max_features=max_features,
                                         max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf,
                                         min_samples_split=min_samples_split,
                                         bootstrap=bootstrap)
        r_forest.fit(x_train, y_train)
        y_pred = r_forest.predict(x_test)
        error = mean_absolute_error(y_test, y_pred)
        print(f"Random Forest Error : {error}")
        all_data.append([n_estimators, max_features, max_depth,
                         min_samples_leaf, min_samples_split, bootstrap, error])


    write_to_csv(data_set, "random_forest",
                 ['n_estimators', 'max_features', 'max_depth', 'min_samples_leaf',
                  'min_samples_split', 'bootstrap', 'Error-'], all_data, True)

def create_lasso_data_set(data_set):
 
    x_train, x_test, y_train, y_test = create_dataframe(data_set)

    #Generating hyperparameter data
    all_data = []
    alpha = 1
    while alpha <= 100:
        for positive in [True, False]:
            for fit_intercept in [True, False]:
                for warm_start in [True, False]:
                    for max_iter in range(300, 5000, 500):
                        for selection in ['cyclic', 'random']:
                            tol = .00000001
                            while tol <= .1:
                                lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept, positive = positive, warm_start=warm_start, selection = selection)
                                lasso.fit(x_train, y_train)
                                y_pred = lasso.predict(x_test)
                                error = mean_absolute_percentage_error(y_test, y_pred)
                                print(f"Error : {error}")
                                all_data.append([alpha, max_iter, tol, fit_intercept, positive, warm_start, selection, error])
                                tol *= 10
        alpha += 5


    write_to_csv(data_set, "lasso",['Alpha', 'Max_iter', 'Tolerance', 'fit_intercept', 'positive', 'warm_start', 'selection', 'Error-'], all_data)

def create_dt_regressor_data_set(data_set):
    x_train, x_test, y_train, y_test = create_dataframe(data_set)

    #Generating hyperparameter data
    all_data = []
    random_state = 3200
    for criterion in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
        for splitter in ['best', 'random']:
            for min_samples_split in [ 2, 5, 10, 37, 73, 100]:
                for min_samples_leaf in [1, 2, 5, 10, 37, 73, 100]:
                    ccp_alpha = 0.0
                    while ccp_alpha <= 1:
                        max_depth = 1
                        while max_depth <= 10000:
                            regressor = DecisionTreeRegressor(random_state=random_state, criterion=criterion, splitter=splitter,  min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, max_depth=max_depth)
                            regressor.fit(x_train, y_train)
                            y_pred = regressor.predict(x_test)
                            error = mean_absolute_percentage_error(y_test, y_pred)
                            print(f"Error : {error}")
                            all_data.append([criterion, splitter, max_depth, error])
                            max_depth *= 10
                        ccp_alpha += 0.1

    write_to_csv(data_set, "decision tree",['criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'max_depth', 'Error-'], all_data)

def create_knn_data_set(data_set):

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    #Generating data
    all_data = []

    for n_neighbors in range(1, 11):
        for weights in ['uniform', 'distance']:
            for algorithm in [ 'ball_tree', 'kd_tree', 'brute']:
                for leaf_size in range(10, 101, 10):
                    for p in np.arange(1, 5, 0.5):
                        for metric in ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']:
                            knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric)
                            knn.fit(x_train, y_train)
                            y_pred = knn.predict(x_test)
                            error = mean_absolute_error(y_test, y_pred)
                            print([n_neighbors, weights, algorithm, leaf_size, p, metric, error])
                            all_data.append([n_neighbors, weights, algorithm, leaf_size, p, metric, error])

    write_to_csv(data_set, "knn", ['N_neighbours', 'weights', 'algorithm', 'Leaf_size', 'P', 'metric', 'Error-'], all_data)


def create_dataframe(data_set):
    data_file = f'../data/{data_set}/{data_set}.csv'

    #createing a dataframe
    df = pd.read_csv(data_file)
    print(df.columns.to_numpy())

    # get the x and y column indexes
    xs, ys =  [], []
    for i, column in enumerate( df.columns.to_numpy()):
        if column.endswith('-') or column.endswith('+'):
            ys.append(i)
        else:
            xs.append(i)

    # #splitting the dataframe into train and test    
    X = df.iloc[:,xs]  
    Y = df.iloc[:,ys]  
    return train_test_split(X, Y, test_size=0.2, random_state=100)

def write_to_csv(data_set, algorithm_name, column_names, data, already_random=False):

    ## five by five
    #shuffle data 5 times first
    for i in range(5):
        random.shuffle(data)

    if not already_random:
        #randomly selecting 10000 rows from the generated data
        selected_data = random.sample(data, min(10000, len(data)))
    else:
        selected_data = data
    #dividing it into 5
    final_data = np.array_split(selected_data, 5)

    # Create the directory (handles non-existent parent directories)
    directory = os.path.dirname(f'../data/{data_set}/{algorithm_name}/')
    os.makedirs(directory, exist_ok=True)

    for i, data in enumerate(final_data):
        file_path = f'../data/{data_set}/{algorithm_name}/{algorithm_name}_hyperparameters_{i + 1}.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            for row in data:
                writer.writerow(row)

# datasets = ['SS-A']
datasets = [ 'Wine_quality', 'pom3a', 'pom3c', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6']
datasets = ['Wine_quality']
for dataset in datasets:
    print(f'-------------------------------------------------------------------------------------------{dataset}-----------------------------------------------------------------------------------------')
    # create_lasso_data_set(dataset)
    #create_dt_regressor_data_set(dataset)
    create_random_forest_regression_data_set(dataset)
