import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, f1_score
import streamlit as st

def tune_random_forest(X, y, n_splits=5):
    """
    Perform hyperparameter tuning for RandomForestClassifier using GridSearchCV with time series split.

    Parameters:
    - X: Features (DataFrame)
    - y: Labels (Series)
    - n_splits: Number of cross-validation splits (TimeSeriesSplit)

    Returns:
    - best_model: Trained model with best parameters
    - best_params: Dictionary of best parameters
    - results_df: DataFrame of all results
    """
    st.subheader("🎯 Tuning Random Forest Hyperparameters")

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(f1_score, average='binary', pos_label=1)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring=scorer,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Display best parameters
    st.write("🔧 Best Parameters:")
    st.json(best_params)

    results_df = (
        pd.DataFrame(grid_search.cv_results_)
        .sort_values(by='mean_test_score', ascending=False)
        .reset_index(drop=True)
    )

    # Display results
    st.dataframe(results_df[['params', 'mean_test_score']])

    return best_model, best_params, results_df