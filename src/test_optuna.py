import csv
import optuna
import random
import time
import util as l

def optuna_for_decision_tree(filename, population):
    with open(filename, mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        random.shuffle(all_data)
        study = optuna.create_study()

        for i in range(1, min(population, len(all_data))):
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
        best_params = study.best_params
        return [ best_params.get('criterion'), best_params.get('splitter'), best_params.get('min_samples_split'), best_params.get('min_samples_leaf'), best_params.get('ccp_alpha'), best_params.get('max_depth'),best_params.get('min_weight_fraction_leaf'), study.best_trial.value]




def optuna_for_lasso(filename, population):
    with open(filename, mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        random.shuffle(all_data)
        study = optuna.create_study()

        for i in range(1, min(population, len(all_data))):
                lines = all_data[i]
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
            # print(study.best_params, study.best_trial.value)    
        best_params = study.best_params
        return [ best_params.get('Alpha'), best_params.get('Max_iter'), best_params.get('Tolerance'), best_params.get('fit_intercept'), best_params.get('positive'), best_params.get('warm_start'), best_params.get('selection'),study.best_trial.value]



def optuna_for_elasticnet(filename, population):
    with open(filename, mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        random.shuffle(all_data)
        study = optuna.create_study()

        for i in range(1, min(population, len(all_data))):
                lines = all_data[i]
                study.add_trial(
                    optuna.trial.create_trial(
                        params={
                            'Alpha' : float(lines[0]),
                            'L1_ratio' : float(lines[1]),
                            'fit_intercept' : lines[2] == 'True',
                            'Max_iter' : int(lines[3]),
                            'selection' : lines[4],
                            'warm_start' : lines[5] == 'True',
                            'Tol' : float(lines[6]),
                        },
                        distributions={
                            'Alpha' : optuna.distributions.FloatDistribution(0, 100),
                            'L1_ratio' : optuna.distributions.FloatDistribution(0, 100),
                            'fit_intercept' : optuna.distributions.CategoricalDistribution([True, False]),
                            'Max_iter' : optuna.distributions.IntDistribution(500, 5000),
                            'selection' : optuna.distributions.CategoricalDistribution(['cyclic', 'random']),
                            'warm_start' : optuna.distributions.CategoricalDistribution([True, False]),
                            'Tol' : optuna.distributions.FloatDistribution(0, 1),
                        },
                        value=float(lines[7]),
                    )
                )
            # print(study.best_params, study.best_trial.value)    
        best_params = study.best_params
        return [ best_params.get('Alpha'), best_params.get('L1_ratio'), best_params.get('fit_intercept'), best_params.get('Max_iter'), best_params.get('selection'), best_params.get('warm_start'), best_params.get('Tol'),study.best_trial.value]




def optuna_for_knn(filename, population):
    with open(filename, mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        random.shuffle(all_data)
        study = optuna.create_study()

        for i in range(1, min(population, len(all_data))):
                lines = all_data[i]
                study.add_trial(
                    optuna.trial.create_trial(
                        params={
                            'N_neighbours' : int(lines[0]),
                            'weights' : lines[1],
                            'algorithm' : lines[2],
                            'Leaf_size' : int(lines[3]),
                            'P' : float(lines[4]),
                            'metric' : lines[5],
                        },
                        distributions={
                            'N_neighbours' : optuna.distributions.IntDistribution(1, 11),
                            'weights' : optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
                            'algorithm' : optuna.distributions.CategoricalDistribution([ 'ball_tree', 'kd_tree', 'brute']),
                            'Leaf_size' : optuna.distributions.IntDistribution(10, 101),
                            'P' : optuna.distributions.FloatDistribution(1, 5),
                            'metric' : optuna.distributions.CategoricalDistribution(['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']),
                        },
                        value=float(lines[6]),
                    )
                )
            # print(study.best_params, study.best_trial.value)    
        best_params = study.best_params
        return [ best_params.get('N_neighbours'), best_params.get('weights'), best_params.get('algorithm'), best_params.get('Leaf_size'), best_params.get('P'), best_params.get('metric'), study.best_trial.value]

def optuna_for_random_forest(filename, population):
    with open(filename, mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        random.shuffle(all_data)
        study = optuna.create_study()
        for i in range(1, min(population, len(all_data))):
            lines = all_data[i]
            study.add_trial(
                optuna.trial.create_trial(
                    params={
                        'N_estimators' : int(lines[0]),
                        'max_features' : None if lines[1] == '' else lines[1],
                        'Max_depth' : float('0.0' if lines[2] == '' else lines[2]),
                        'Min_samples_leaf' : int(lines[3]),
                        'Min_samples_split' : int(lines[4]),
                        'bootstrap' : lines[5] == 'True',
                    },
                    distributions={
                        'N_estimators' : optuna.distributions.IntDistribution(5, 800),
                        'max_features' : optuna.distributions.CategoricalDistribution([None, "sqrt", "log2"]),
                        'Max_depth' : optuna.distributions.FloatDistribution(0, 300),
                        'Min_samples_leaf' : optuna.distributions.IntDistribution(1, 50),
                        'Min_samples_split' : optuna.distributions.IntDistribution(2, 20),
                        'bootstrap' : optuna.distributions.CategoricalDistribution([True, False]),
                    },
                    value=float(lines[6]),
                )
            )   
        best_params = study.best_params
        return [ best_params.get('N_estimators'), best_params.get('max_features'), best_params.get('Max_depth'), best_params.get('Min_samples_leaf'), best_params.get('Min_samples_split'), best_params.get('bootstrap'), study.best_trial.value]


"""optuna_for_random_forest('Wine_quality', 1000)
optuna_for_decision_tree('Wine_quality', 1000)
optuna_for_knn('Wine_quality', 1000)
optuna_for_lasso('Wine_quality', 1000)
optuna_for_elasticnet('Wine_quality', 1000)"""
