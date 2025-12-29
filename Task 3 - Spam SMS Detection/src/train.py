from sklearn.model_selection import GridSearchCV
from config import SCORING, CV_FOLDS

def train_models(X_train, y_train, model_configs):
    results = {}

    for name, config in model_configs.items():
        print(f"\nTUNING {name}")
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            scoring=SCORING,
            cv=CV_FOLDS,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        results[name] = grid

        print("Best F1:", grid.best_score_)
        print("Best Params:", grid.best_params_)

    return results
