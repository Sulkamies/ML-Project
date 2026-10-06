from data_split import dataImport, trainTestSplit, foldSplit
from validationTraining import gridSearch
from training import train

# Hyperparameters
lamda = 0.1
k = 250

# Import raw (but normalized) data
X_raw, y = dataImport()

# Performs split into train and test sets 
X_train_raw, X_test_raw, y_train, y_test = trainTestSplit(X_raw, y)

# Train the model using the hyperparameters chosen
train(X_train_raw, y_train, X_test_raw, y_test, k, lamda)

# For testing

#print(X_train_compressed.shape)
#print(X_test_compressed.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(X_inverse.shape)

