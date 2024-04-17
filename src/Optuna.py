import optuna
import utils
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


datasets = ['pom3a', 'pom3c']
for dataset in datasets:
    data = []

    def objective(trial):
        x_train, x_test, y_train, y_test = utils.create_dataframe(dataset)
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
        data.append([criterion, splitter, min_samples_split, min_samples_leaf, ccp_alpha, max_depth, error])
        return error
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)
    utils.wrtie_to_csv(dataset, "optuna",['criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'max_depth', 'Error-'], data)
