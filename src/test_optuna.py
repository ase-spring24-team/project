import csv
import optuna

def optuna_for_decision_tree():
    for i in range(1, 6):
        study = optuna.create_study()
        with open(f'../data/SS-A/decision tree/decision tree_hyperparameters_{i}.csv', mode ='r')as file:
            csv_file = csv.reader(file)
            first_line = True
            for lines in csv_file:
                    if first_line:
                        first_line = False
                        continue
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
            print(study.best_params)


def optuna_for_lasso():
    for i in range(1, 6):
        study = optuna.create_study()
        with open(f'../data/SS-A/lasso/lasso_hyperparameters_{i}.csv', mode ='r')as file:
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
            print(study.best_params)    

# optuna_for_decision_tree()
optuna_for_lasso()