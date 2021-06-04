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
	test_dict = {}
	coeffs = {}
	pred_probs = {}
	df = X.copy()
	y, X = np.array(y), np.array(X)
	for train_indices, test_indices in kf.split(X, y):
		train_scaler = StandardScaler().fit(X[train_indices])
		X_train = pd.DataFrame(train_scaler.transform(X[train_indices]))
		y_train = y[train_indices]
		X_test  = pd.DataFrame(train_scaler.transform(X[test_indices]))
		y_test  = y[test_indices]
		lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
								max_iter=1000, random_state=12, n_jobs=8).fit(X_train, y_train)
		y_pred = lr.predict_proba(X_test)
		pred_probs[fold] = [x[1] for x in y_pred]
		y_pred = [1 if x[1] > 0.25 else 0 for x in y_pred]
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
	prediction_df = pd.DataFrame.from_dict(pred_probs, orient='index').transpose()
	prediction_df['avg'] = prediction_df.mean(axis=1)
	df['Average Prediction'] = prediction_df['avg']
	df['True Slide'] = y
	# Ys = [model.predict_proba([[value]])[0][1] for value in range(x_range)]

	# plt.scatter(df['X'], df['y'])
	# plt.plot(Xs, Ys, color='red')
	# prediction_df = pd.DataFrame.from_dict(pred_probs,orient='index').transpose()
	# prediction_df = pd.DataFrame(list(pred_probs.values()), columns=list(pred_probs.keys())) 
	df.to_csv('../data/model_output/logit_predictions.csv')


def load_data():
	df = pd.read_csv('../data/final_landslides.csv', index_col=0)
	y, X = df['true_slide'], df.drop('true_slide', 1)
	return y, X

if __name__ == '__main__':
	main()