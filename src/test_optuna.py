import csv
import optuna
import utils
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def objective(trial):
    x_train, x_test, y_train, y_test = utils.create_dataframe('SS-A')
    random_state = 3200
    criterion = trial.suggest_categorical( 'criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
    splitter = trial.suggest_categorical('splitter', ['best', 'random'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    ccp_alpha = trial.suggest_float('ccp_alpha', 0, 1)
    max_depth = trial.suggest_int('max_depth', 1, 10000)
    regressor = DecisionTreeRegressor(random_state=random_state, criterion=criterion, splitter=splitter,  min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, max_depth=max_depth)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    error = mean_absolute_error(y_test, y_pred)
    return error

study = optuna.create_study()


with open('../data/SS-A/decision tree/decision tree_hyperparameters_1.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  firstLine = True
  for lines in csvFile:
        if firstLine:
            firstLine = False
            continue
        print(lines)
        study.add_trial(
            optuna.trial.create_trial(
                params={
                    'criterion' : lines[0],
                    'splitter' : lines[1],
                    'min_samples_split' : int(lines[2]),
                    'min_samples_leaf' : int(lines[3]),
                    'ccp_alpha' : float(lines[4]),
                    'max_depth' : int(lines[5]),
                },
                distributions={
                    'criterion' : optuna.distributions.CategoricalDistribution(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                    'splitter' : optuna.distributions.CategoricalDistribution(['best', 'random']),
                    'min_samples_split' : optuna.distributions.IntDistribution(2, 100),
                    'min_samples_leaf' : optuna.distributions.IntDistribution(1, 100),
                    'ccp_alpha' : optuna.distributions.FloatDistribution(0, 1),
                    'max_depth' : optuna.distributions.IntDistribution(1, 10000),
                },
                value=float(lines[6]),
            )
        )
study.optimize(objective, n_trials=2)
print(study.best_params)