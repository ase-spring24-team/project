#importing the required packages
import csv
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def create_lasso_data_set(data_file):
    #createing a dataframe
    df = pd.read_csv(data_file)
    df_binary = df[['Spout_wait', 'Spliters', 'Counters', 'Throughput+', 'Latency-']]
    df_binary.columns = ['Wait', 'Split', 'Count', 'Throughput', 'Latency']

    #splitting the dataframe into train and test
    X = df_binary.iloc[:, :3]  
    Y = df_binary.iloc[:, 3:]  
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


    selected_data = random.sample(all_data, 10000)
    final_data = np.array_split(selected_data, 5)

    # Writing to CSV
    for i, data in enumerate(final_data):
        file_path = '../data/lasso_hyperparameters_' + str(i + 1) + '.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Alpha', 'Max_iter', 'Tolerance', 'fit_intercept', 'positive', 'warm_start', 'selection', 'Error-'])
            for row in data:
                writer.writerow(row)


create_lasso_data_set('../data/SS-A.csv')
