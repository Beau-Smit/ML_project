import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, roc_auc_score
from xgboost import plot_tree
import matplotlib.pyplot as plt

def main():
	y, X = load_data()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
	test_dict = {}
	df = X.copy()
	y, X = np.array(y), np.array(X)
	feature_importances = {}
	fold = 0
	for train_indices, test_indices in kf.split(X, y):
		train_scaler = StandardScaler().fit(X[train_indices])
		X_train = pd.DataFrame(train_scaler.transform(X[train_indices]))
		y_train = y[train_indices]
		X_test  = pd.DataFrame(train_scaler.transform(X[test_indices]))
		y_test  = y[test_indices]

		xgb = XGBClassifier(n_estimators=100, eval_metric=['logloss'], n_jobs=8, 
							random_state=12, booster='gbtree', use_label_encoder=False)
		fitted_xgb = xgb.fit(X_train, y_train)
		y_pred = fitted_xgb.predict_proba(X_test)
		y_pred = [1 if x[1] > 0.04 else 0 for x in y_pred]
		test_dict[fold] = [log_loss(y_test, y_pred), accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred), 
						   precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
		feature_importances[fold] = fitted_xgb.feature_importances_
		fold += 1		
		plot_tree(fitted_xgb, num_trees=5)
		fig = plt.gcf()
		fig.set_size_inches(150, 100)
		fig.savefig('../img/tree_plots/tree_'+str(fold)+'.png')
	test_df = pd.DataFrame(test_dict).T
	test_df.columns = ['Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
	test_df.to_csv('../data/model_output/xgb_test_data.csv')
	fi_df = pd.DataFrame.from_dict(feature_importances, orient='index', columns=df.columns.tolist())
	mean_df  = pd.DataFrame({col: fi_df[col].mean() for col in fi_df.columns.tolist()}, index=['avg'])
	fi_df = pd.concat([fi_df, mean_df])
	fi_df.to_csv('../data/model_output/xgb_feature_importances.csv')


def load_data():
	df = pd.read_csv('../data/final_landslides.csv', index_col=0)
	y, X = df['true_slide'], df.drop('true_slide', 1)
	return y, X

if __name__ == '__main__':
	main()