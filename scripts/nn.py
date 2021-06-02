import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Precision, Recall
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def main():
	y, X = load_data()
	kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
	training_dict = {}
	test_dict = {}
	fold = 1
	total_epoch_count = 0
	for train_indices, test_indices in kf.split(X, y):
		X_train = pd.DataFrame(StandardScaler().fit_transform(X[train_indices]))
		y_train = y[train_indices]
		X_test  = pd.DataFrame(StandardScaler().fit_transform(X[test_indices]))
		y_test  = y[test_indices]
		model = Sequential()
		model.add(Dense(17, input_dim=17, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Recall(), Precision(), 'AUC'])
		history = model.fit(X_train, y_train, batch_size=32, epochs=50, workers=8, use_multiprocessing=True, validation_data=(X_test, y_test))
		# print(history.history.keys())
		# sys.exit()
		for epoch_count in range(50):
			index = (fold-1)*50+epoch_count
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
		fold += 1
	training_data = pd.DataFrame(training_dict).T
	training_data.columns = ['Fold Number', 'Epoch', 'Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall', 
							 'Validation Loss', 'Validation Accuracy', 'Validation Recall', 'Validation Precision', 'Validation AUC']
	test_data = pd.DataFrame(test_dict).T
	test_data.columns = ['Binary Cross-Entropy Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']

	training_data.to_csv('../data/model_output/nn_training_data.csv')
	test_data.to_csv('../data/model_output/nn_test_data.csv')


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