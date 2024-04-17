"""
File by Sathiya Narayanan Venkatesan
This file contains the main functions of the project
"""


#importing the required packages

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from Data import Data
import Sample
from datetime import date
import util as l
from the import the, SLOTS
import random
from statistics import mean, stdev

def create_elasticnet_data_set(data_set):

    x_train, x_test, y_train, y_test = l.create_dataframe(data_set)

    #Generating data
    all_data = []

    for i in range(10000):
        # loop through and create 10000 random hyper sets
        alpha = random.randint(1, 100)*.1
        l1_ratio = random.randint(0, 100)*.01
        fit_intercept = random.sample([True, False], 1)[0]
        max_iter = random.randint(500, 5000)
        selection = random.sample(['cyclic', 'random'], 1)[0]
        warm_start = random.sample([True, False], 1)[0]
        tol = random.uniform(0.000000001, 0.000001)

        elasticnet = ElasticNet(random_state=0,
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                fit_intercept=fit_intercept,
                                max_iter=max_iter,
                                selection=selection,
                                warm_start=warm_start,
                                tol=tol)

        elasticnet.fit(x_train, y_train)
        y_pred = elasticnet.predict(x_test)
        error = mean_absolute_error(y_test, y_pred)
        print(f"ElasticNet Error : {error}")
        all_data.append([alpha, l1_ratio, fit_intercept,
                         max_iter, selection, warm_start, tol, error])

    l.write_to_csv(data_set, "ElasticNet",
                 ['Alpha', 'L1_ratio', 'fit_intercept', 'Max_iter', 'selection',
                  'warm_start', 'Tol', 'Error-'], all_data, True)

def create_random_forest_regression_data_set(data_set):

    x_train, x_test, y_train, y_test = l.create_dataframe(data_set)

    #Generating data
    all_data = []

    for i in range(10000):
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


    l.write_to_csv(data_set, "random_forest",
                 ['n_estimators', 'max_features', 'max_depth', 'min_samples_leaf',
                  'min_samples_split', 'bootstrap', 'Error-'], all_data, True)

def create_lasso_data_set(data_set):
 
    x_train, x_test, y_train, y_test = l.create_dataframe(data_set)

    #Generating hyperparameter data
    all_data = []
    alpha = 1
    while alpha <= 100:
        for positive in [True, False]:
            for fit_intercept in [True, False]:
                for warm_start in [True, False]:
                    for max_iter in range(100, 10000, 500):
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


    l.write_to_csv(data_set, "lasso",['Alpha', 'Max_iter', 'Tolerance', 'fit_intercept', 'positive', 'warm_start', 'selection', 'Error-'], all_data)

def create_dt_regressor_data_set(data_set):
    x_train, x_test, y_train, y_test = l.create_dataframe(data_set)

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
                            all_data.append([criterion, splitter, min_samples_split, min_samples_leaf, ccp_alpha, max_depth, error])
                            max_depth *= 10
                        ccp_alpha += 0.1

    l.write_to_csv(data_set, "decision tree",['criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'max_depth', 'Error-'], all_data)

def create_knn_data_set(data_set):

    x_train, x_test, y_train, y_test = l.create_dataframe(data_set)

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

    l.write_to_csv(data_set, "knn", ['N_neighbours', 'weights', 'algorithm', 'Leaf_size', 'P', 'metric', 'Error-'], all_data)

def print_ranking_analysis(d):
    """
    Prints out the ranking analysis
    """
    print("Starting ranking analysis...")
    # found date code at https://www.programiz.com/python-programming/datetime/current-datetime
    today = date.today()
    todays_date = today.strftime("%B %d, %Y")
    print(f"Date : {todays_date}")  # print current date
    print(f"File : {the.file}")  # print file name
    print(f"Repeats : 20")  # print the number of repetitions(num of times we run bonr15
    # when building our sampling group for example)
    print(f"Seed : {the.seed}")
    print(f"Rows : {len(d.rows)}")
    print(f"Columns : {len(d.cols.all)}")

def get_best_bonr(num):
    """
    Runs bonrN once and returns the best d2h value found
    """
    d = Data(the.file)
    _stats, _bests = d.gate(4, num-4, .5, False) # bonr9 if num = 9, bonr15 if num = 15 etc.
    # I also added a parameter above so that we don't have to always print all the baselines
    # when running gate
    stat, best = _stats[-1], _bests[-1]
    #print(best.d2h(d))
    #print(_bests[0].d2h(d))
    assert best.d2h(d) <= _bests[0].d2h(d)  # Tests that we are getting the best value based on d2h
    # and not some other value by accident
    return l.rnd(best.d2h(d))

def get_best_rrp(num=None):
    """
    Runs rrpN once and returns the best d2h value found
    """
    d = Data(the.file)
    best, rest, evals = d.branch(num)  # num is the stop number (in regular rrp, the num is half
    best.rows.sort(key=lambda x: x.d2h(d))
    # the length of the data set
    return l.rnd(best.rows[0].d2h(d))

def get_best_rrpDT():
    """
    Runs rrp double tap
    """
    d = Data(the.file)
    best1, rest, evals1 = d.branch(32)
    best2, _, evals2 = best1.branch(4)
    best2.rows.sort(key=lambda x: x.d2h(d))
    # the length of the data set
    return l.rnd(best2.rows[0].d2h(d))

def get_best_rand(num):
    """
    Runs randN once and returns the best d2h value found for the sample of num numbers
    """
    d = Data(the.file)
    rows = random.sample(d.rows, num)  # sample N number of random rows
    rows.sort(key=lambda x: x.d2h(d))  # sort the rows by d2h and pull out the best value
    return l.rnd(rows[0].d2h(d))  # return the d2h of the best row

def get_base_line_list(rows,d):
    """
    Takes a list of all rows in the data set d, and returns a list of all row's d2h values
    :param rows: list of all rows in data d
    """
    d2h_list = []
    for row in rows:
        d2h_list.append(row.d2h(d))
    return d2h_list

def ranking_stats():
    """
    Runs smo, rrp, optuna, and hyperband and compares them all to each other
    """
    d = Data(the.file)  # just set d for easy use in print statements
    print_ranking_analysis(d)
    all_rows = d.rows
    # Now we must sort all rows based on the distance to heaven to get our ceiling
    all_rows.sort(key=lambda x: x.d2h(d))
    ceiling = l.rnd(all_rows[0].d2h(d))  # set ceiling value to best value
    bonr9_best_list = []  # the list of 20 best bonr9 value
    rand9_best_list = []  # the list of 20 best rand9 value
    bonr15_best_list = []
    rand15_best_list = []
    bonr20_best_list = []
    rand20_best_list = []
    rrp_best_list = []
    rrp_doubletap_best_list = []
    rand358_best_list = []
    print("Calculating Best and Tiny...")
    for i in range(20):
        # iterate our 20 times
        bonr9_best_list.append(get_best_bonr(9))  # calls to a function that runs data for bonr9
        # and returns the best value once
        rand9_best_list.append(get_best_rand(9))  # calls to function which randomly samples
        # 9 rows from the data set and returns the best rows d2h
        bonr15_best_list.append(get_best_bonr(15))
        rand15_best_list.append(get_best_rand(15))
        bonr20_best_list.append(get_best_bonr(20))
        rand20_best_list.append(get_best_rand(20))
        rrp_best_list.append(get_best_rrp())
        rrp_doubletap_best_list.append(get_best_rrpDT())
        rand358_best_list.append(get_best_rand(358))
    base_line_list = get_base_line_list(d.rows, d)  # returns a list of all rows d2h values
    std = stdev(base_line_list)  # standard deviation of all rows d2h values
    print(f"Best : {ceiling}")  #
    print(f"Tiny : {l.rnd(.35*std)}")  # WE NEED to change this later...

    print("base bonr9 rrp9 rand9 bonr15 rand15 bonr20 rand20 rrp rrpDT rand358")
    print("Ranking Report: ")
    #  Below is the code that will actually stratify and print the different treatments
    Sample.eg0([
        Sample.SAMPLE(bonr9_best_list, "bonr9"),
        Sample.SAMPLE(rand9_best_list, "rand9"),
        Sample.SAMPLE(bonr15_best_list, "bonr15"),
        Sample.SAMPLE(rand15_best_list, "rand15"),
        Sample.SAMPLE(bonr20_best_list, "bonr20"),
        Sample.SAMPLE(rand20_best_list, "rand20"),
        Sample.SAMPLE(rrp_best_list, "rrp"),
        Sample.SAMPLE(rrp_doubletap_best_list, "rrpDT"),
        Sample.SAMPLE(rand358_best_list, "rand358"),
        Sample.SAMPLE(base_line_list, "base"),
    ])

if __name__ == '__main__':
    the._set(SLOTS({"file":"../data/dtlz2/random_forest/random_forest_hyperparameters_1.csv", "__help": "", "m":2, "k":1, "p":2, "Half":256, "d":32, "D":4,
                    "Far":.95, "seed":31210, "Beam":10, "bins":16, "Cut":.1, "Support":2}))
    random.seed(the.seed)
    datasets = ['SS-A']
    # datasets = [ 'Wine_quality', 'pom3a', 'pom3c', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'SS-A', 'SS-K']
    for dataset in datasets:
        print(f'-------------------------------------------------------------------------------------------{dataset}-----------------------------------------------------------------------------------------')
        # create_lasso_data_set(dataset)
        create_dt_regressor_data_set(dataset)
        #create_random_forest_regression_data_set(dataset)
        #create_elasticnet_data_set(dataset)
        # ranking_stats()  # runs on the.file currently
