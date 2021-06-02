import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Precision, Recall
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	y, X = load_data()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
	training_dict = {}
	test_dict = {}
	comparable_test_dict = {}
	roc_data = {}
	fold = 1
	total_epoch_count = 0
	for train_indices, test_indices in kf.split(X, y):
		train_scaler = StandardScaler().fit(X[train_indices])
		X_train = pd.DataFrame(train_scaler.transform(X[train_indices]))
		y_train = y[train_indices]
		X_test  = pd.DataFrame(train_scaler.transform(X[test_indices]))
		y_test  = y[test_indices]
		model = Sequential()
		model.add(Dense(17, input_dim=17, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(17, input_dim=17, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Recall(), Precision(), 'AUC'])
		history = model.fit(X_train, y_train, batch_size=32, epochs=100, workers=8, use_multiprocessing=True, validation_data=(X_test, y_test))
		for epoch_count in range(100):
			index = (fold-1)*100+epoch_count
			if 'precision' in history.history.keys():
				training_dict[index] = [fold, epoch_count+1, history.history['loss'][epoch_count], history.history['accuracy'][epoch_count], 
										history.history['auc'][epoch_count], history.history['precision'][epoch_count], 
										history.history['recall'][epoch_count], history.history['val_loss'][epoch_count], 
										history.history['val_accuracy'][epoch_count], history.history['val_recall'][epoch_count],
										history.history['val_precision'][epoch_count], history.history['val_auc'][epoch_count]]
			else:
				training_dict[index] = [fold, epoch_count+1, history.history['loss'][epoch_count], history.history['accuracy'][epoch_count], 
										history.history['auc'][epoch_count], history.history['precision_'+str(fold-1)][epoch_count], 
										history.history['recall_'+str(fold-1)][epoch_count], history.history['val_loss'][epoch_count], 
										history.history['val_accuracy'][epoch_count], history.history['val_recall_'+str(fold-1)][epoch_count],
										history.history['val_precision_'+str(fold-1)][epoch_count], history.history['val_auc'][epoch_count]]
		test_dict[fold] = model.evaluate(X_test, y_test, batch_size=32, workers=8, use_multiprocessing=True)
		y_pred = model.predict(X_test, workers=8, use_multiprocessing=True).flatten()
		y_pred_actual = y_pred.copy()
		y_pred = [1 if x > 0.25 else 0 for x in y_pred]
		comparable_test_dict[fold] = [log_loss(y_test, y_pred), accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred), 
						   			  precision_score(y_test, y_pred), recall_score(y_test, y_pred)]

		fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_actual)
		roc_data[fold] = [fpr, tpr]
		fold += 1
	training_data = pd.DataFrame(training_dict).T
	training_data.columns = ['Fold Number', 'Epoch', 'Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall', 
							 'Validation Loss', 'Validation Accuracy', 'Validation Recall', 'Validation Precision', 'Validation AUC']
	test_data = pd.DataFrame(test_dict).T
	test_data.columns = ['Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']

	comparable_test_df = pd.DataFrame.from_dict(comparable_test_dict, orient='index')
	comparable_test_df.columns = ['Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']

	training_data.to_csv('../data/model_output/nn_training_data.csv')
	test_data.to_csv('../data/model_output/nn_test_data.csv')
	comparable_test_df.to_csv('../data/model_output/nn_comparable_test_data.csv')

	plot_roc(roc_data)


def plot_roc(roc_data):
	f = plt.figure(figsize=(15,15))
	ax1 = f.add_subplot(111)
	ax1.plot([0,1], [0, 1], 'k--')
	ax1.set_xlabel('False positive rate')
	ax1.set_ylabel('True positive rate')
	ax1.set_title('ROC Curves for Neural Network by Fold')
	for fold in roc_data.keys():
		fpr = roc_data[fold][0]
		tpr = roc_data[fold][1]
		ax1.plot(fpr, tpr, label=f'Fold {fold}')
	plt.savefig('../img/nn_roc.png')




def load_data():
	df = pd.read_csv('../data/final_landslides.csv', index_col=0)
	y, X = df['true_slide'], df.drop('true_slide', 1)
	y, X = np.array(y), np.array(X)
	return y, X


# def build_nn():
# 	# There are 17 columns in the dataset
# 	model = Sequential()
# 	model.add(Dense(17, input_dim=17, activation='relu'))
# 	model.add(Dense(8, activation='relu'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
# 	return model


if __name__ == '__main__':
	main()