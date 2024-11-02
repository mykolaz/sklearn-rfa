# sklearn-rfa
Recursive Feature Addition for testing against sklearn.feature_selection's RFE(_rfe.py)

## Usage

params = {
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'learning_rate': LEARNING_RATE,
        'subsample': SUBSAMPLE,
        'colsample_bytree': SAMPLES_BY_TREE,
        'colsample_bylevel': SAMPLES_BYLEVEL,
        'colsample_bynode': SAMPLES_BY_NODE,
        'reg_alpha': ALPHA,
        'reg_lambda': LAMBDA,
        'gamma': GAMMA,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'booster': 'gbtree',
        'min_child_weight': MIN_CHILD_WEIGHT,
        'enable_categorical': True
    }

model = xgb.XGBRegressor(
    **params
    )
rfacv = RFACV(cv=8, model=model)
rfacv.fit(X, y)

print("\nSelected Features:", rfacv.get_selected_features())
print("Best CV Score:", rfacv.get_best_score())

rfacv.plot_scores()

## Results
