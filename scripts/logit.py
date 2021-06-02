import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, roc_auc_score


def main():
	y, X = load_data()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
	fold = 1
	total_epoch_count = 0
	test_dict = {}
	coeffs = {}
	df = X.copy()
	y, X = np.array(y), np.array(X)
	for train_indices, test_indices in kf.split(X, y):
		X_train = pd.DataFrame(StandardScaler().fit_transform(X[train_indices]))
		y_train = y[train_indices]
		X_test  = pd.DataFrame(StandardScaler().fit_transform(X[test_indices]))
		y_test  = y[test_indices]
		lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
								max_iter=1000, random_state=12, n_jobs=8).fit(X_train, y_train)
		y_pred = lr.predict(X_test)
		coeffs[fold] = lr.coef_[0]
		test_dict[fold] = [log_loss(y_test, y_pred), accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred), 
						   precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
		fold += 1
	test_df = pd.DataFrame(test_dict).T
	test_df.columns = ['Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
	test_df.to_csv('../data/model_output/logit_test_data.csv')
	coeff_df = pd.DataFrame.from_dict(coeffs, orient='index', columns=df.columns.tolist())
	mean_df  = pd.DataFrame({col: coeff_df[col].mean() for col in coeff_df.columns.tolist()}, index=['avg'])
	coeff_df = pd.concat([coeff_df, mean_df])
	coeff_df.to_csv('../data/model_output/logit_coefficients.csv')

def load_data():
	df = pd.read_csv('../data/final_landslides.csv', index_col=0)
	y, X = df['true_slide'], df.drop('true_slide', 1)
	return y, X

if __name__ == '__main__':
	main()