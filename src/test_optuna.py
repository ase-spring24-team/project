import csv
import optuna
import random
import time
import util as l

def optuna_for_decision_tree(dataset):
    with open(f'../data/{dataset}/decision tree/merged_hyperparameters.csv', mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        output = []
        for size in [9, 15, 50, 10000]:
            random.shuffle(all_data)
            study = optuna.create_study()
            for i in range(1, size):
                start_time = time.time()
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
            end_time = time.time()
            total_time = end_time - start_time
            best_params = study.best_params
            output.append([size, best_params.get('criterion'), best_params.get('splitter'), best_params.get('min_samples_split'), best_params.get('min_samples_leaf'), best_params.get('ccp_alpha'), best_params.get('max_depth'), total_time, study.best_trial.value])
            # print(study.best_params, study.best_trial.value)
        # l.write_to_csv(output, "decision tree",['criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'max_depth', 'Clock_speed-', 'Error-'], all_data)   
        print(output)


def optuna_for_lasso(dataset):
    with open(f'../data/{dataset}/lasso/merged_hyperparameters.csv', mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        output = []
        for size in [9, 15, 50, 10000]:
            random.shuffle(all_data)
            study = optuna.create_study()
            for i in range(1, size):
                start_time = time.time()
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
            end_time = time.time()
            total_time = end_time - start_time
            best_params = study.best_params
            output.append([size, best_params.get('Alpha'), best_params.get('Max_iter'), best_params.get('Tolerance'), best_params.get('fit_intercept'), best_params.get('positive'), best_params.get('warm_start'), best_params.get('selection'), total_time, study.best_trial.value])
        print(output)


def optuna_for_elasticnet(dataset):
    with open(f'../data/{dataset}/ElasticNet/merged_hyperparameters.csv', mode ='r')as file:
        csv_file = csv.reader(file)
        all_data = []
        first_line = True
        for row in csv_file:
            if first_line:
                first_line = False
                continue
            all_data.append(row)
        output = []
        for size in [9, 15, 50, 10000]:
            random.shuffle(all_data)
            study = optuna.create_study()
            for i in range(1, size):
                start_time = time.time()
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
            end_time = time.time()
            total_time = end_time - start_time
            best_params = study.best_params
            output.append([size, best_params.get('Alpha'), best_params.get('L1_ratio'), best_params.get('fit_intercept'), best_params.get('Max_iter'), best_params.get('selection'), best_params.get('warm_start'), best_params.get('Tol'), total_time, study.best_trial.value])
        print(output)


# optuna_for_decision_tree('Wine_quality')
optuna_for_lasso('Wine_quality')
# optuna_for_elasticnet('Wine_quality')
