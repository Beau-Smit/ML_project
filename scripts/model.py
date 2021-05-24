import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib

STATE = 846318

df = pd.read_pickle('../data/toy_data.pkl')
X_train, X_test, y_train, y_test = train_test_split(df[['gazetteer_distance']], df[['landslide']], test_size = .2, random_state = STATE)

# support vector
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# sv_params = {'sv__C': np.logspace(-4, 4, 8)} # distribution: 0 - 1000 (mostly near 0)
# sv_gs = GridSearchCV(SVC(), parameters)
# sv_gs.fit(X_train, y_train)

# random forest
# parameters = {'n_estimators':[100, 250, 500]}
# rf_gs = GridSearchCV(RandomForestClassifier(), param_grid, random_state=STATE)
# rf_gs.fit(X_train, y_train)

# Randomized Parameter Optimization
rf_pipe = Pipeline([("rf", RandomForestClassifier(random_state=STATE))])
rf_params = {"rf__min_samples_leaf": np.random.randint(3, 10, 3),
        "rf__n_estimators": np.random.randint(100, 1000, 3),
        "rf__max_features": [None],
        "rf__max_depth": [None, 10]}
rf_rgs = RandomizedSearchCV(rf_pipe, rf_params, n_iter=10, cv=5)

# rf_rgs.fit(X_train, y_train)
# joblib.dump(rf_rgs, '../data/RF_randomized_gs.pkl')
rf_rgs = joblib.load('../data/RF_randomized_gs.pkl')

print(rf_rgs.best_estimator_)

y_preds = rf_rgs.best_estimator_.predict(X_test)
y_probs = rf_rgs.best_estimator_.predict_proba(X_test)[:,1]

acc = (y_preds == y_test['landslide']).sum() / len(y_preds)
print(f'Accuracy = {acc}')
