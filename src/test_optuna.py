import csv
import optuna
import random

def optuna_for_decision_tree(dataset, size):
    for i in range(1, 6):
        study = optuna.create_study()
        with open(f'../data/{dataset}/decision tree/decision tree_hyperparameters_{i}.csv', mode ='r')as file:
            csv_file = csv.reader(file)
            all_data = []
            first_line = True
            for row in csv_file:
                if first_line:
                    first_line = False
                    continue
                all_data.append(row)
            random.shuffle(all_data)
            for i in range(1, size):
                lines = all_data[i]
                study.add_trial(
                    optuna.trial.create_trial(
                        params={
                            'criterion' : lines[0],
                            'splitter' : lines[1],
                            'min_samples_split' : int(lines[2]),
                            'min_samples_leaf' : int(lines[3]),
                            'ccp_alpha' : float(lines[4]),
                            'max_depth' : int(lines[5]),
                            'min_weight_fraction_leaf' : float(lines[6]),
                        },
                        distributions={
                            'criterion' : optuna.distributions.CategoricalDistribution(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                            'splitter' : optuna.distributions.CategoricalDistribution(['best', 'random']),
                            'min_samples_split' : optuna.distributions.IntDistribution(2, 100),
                            'min_samples_leaf' : optuna.distributions.IntDistribution(1, 100),
                            'ccp_alpha' : optuna.distributions.FloatDistribution(0, 1),
                            'max_depth' : optuna.distributions.IntDistribution(1, 10000),
                            'min_weight_fraction_leaf' : optuna.distributions.FloatDistribution(0, 1),
                        },
                        value=float(lines[7]),
                    )
                )
            print(study.best_params, study.best_trial.value)


def optuna_for_lasso(dataset):
    for i in range(1, 6):
        study = optuna.create_study()
        with open(f'../data/{dataset}/lasso/lasso_hyperparameters_{i}.csv', mode ='r')as file:
            csv_file = csv.reader(file)
            first_line = True
            for lines in csv_file:
                    if first_line:
                        first_line = False
                        continue
                    study.add_trial(
                        optuna.trial.create_trial(
                            params={
                                'Alpha' : int(lines[0]),
                                'Max_iter' : int(lines[1]),
                                'Tolerance' : float(lines[2]),
                                'fit_intercept' : lines[3] == 'True',
                                'positive' : lines[4] == 'True',
                                'warm_start' : lines[5] == 'True',
                                'selection' : lines[6],
                            },
                            distributions={
                                'Alpha' : optuna.distributions.IntDistribution(1, 100),
                                'Max_iter' : optuna.distributions.IntDistribution(1, 10000),
                                'Tolerance' : optuna.distributions.FloatDistribution(0, 1),
                                'fit_intercept' : optuna.distributions.CategoricalDistribution([True, False]),
                                'positive' : optuna.distributions.CategoricalDistribution([True, False]),
                                'warm_start' : optuna.distributions.CategoricalDistribution([True, False]),
                                'selection' : optuna.distributions.CategoricalDistribution(['cyclic', 'random']),
                            },
                            value=float(lines[7]),
                        )
                    )
            print(study.best_params, study.best_trial.value)    

# optuna_for_decision_tree('Wine_quality', 2)
# optuna_for_lasso()

# import pandas as pd

# def combine_datasets():
#     # List to store your dataframes
#     datasets = ['Wine_quality', 'pom3a', 'pom3c', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'SS-K']
#     ml_algos = ['random_forest', 'lasso', 'knn', 'ElasticNet', 'decision tree']
#     for data_set in datasets:
#         for algorithm_name in ml_algos:
#             dataframes = []
#             for i in range(1, 6):
#                 filename = f'../data/{data_set}/{algorithm_name}/{algorithm_name}_hyperparameters_{i}.csv'
#                 print(filename)
#                 df = pd.read_csv(filename)
#                 dataframes.append(df)
#             combined_df = pd.concat(dataframes, ignore_index=True)
#             combined_df.to_csv(f'../data/{data_set}/{algorithm_name}/merged_hyperparameters.csv', index=False)

# combine_datasets()
