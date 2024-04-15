"""
File by Sathiya Narayanan Venkatesan
This file contains the main functions of the project
"""


#importing the required packages
import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def create_lasso_data_set(data_set):

    data_file = f'../data/{data_set}/{data_set}.csv'

    #createing a dataframe
    df = pd.read_csv(data_file)
    print(df.columns.to_numpy())
    count = 0
    for column in df.columns.to_numpy():
        if column.endswith('-') or column.endswith('+'):
            break
        count += 1

    # #splitting the dataframe into train and test
    X = df.iloc[:, :count]  
    Y = df.iloc[:, count:]  
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    #Generating data
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


    wrtie_to_csv(data_set, "lasso",['Alpha', 'Max_iter', 'Tolerance', 'fit_intercept', 'positive', 'warm_start', 'selection', 'Error-'], all_data)


def wrtie_to_csv(data_set, algorithm_name, column_names, data):

    ## five by five
    #randomly selecting 10000 rows from the generated data
    selected_data = random.sample(data, 10000)
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

datasets = ['SS-A', 'SS-N', 'Wine_quality', 'pom3a', 'pom3c', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6']
for dataset in datasets:
    print(f'-------------------------------------------------------------------------------------------{dataset}-----------------------------------------------------------------------------------------')
    create_lasso_data_set(dataset)
