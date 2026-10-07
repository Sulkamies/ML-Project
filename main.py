from data_split import dataImport, trainTestSplit, foldSplit
from validationTraining import gridSearch
from training import train
from logistic_regressor import train_logistic_regressor, train_once
from sklearn.metrics import log_loss, accuracy_score


# Hyperparameters
lamda = 0.1
k = 250

# Import raw (but normalized) data
X_raw, y = dataImport()

# Performs split into train and test sets 
X_train_raw, X_test_raw, y_train, y_test = trainTestSplit(X_raw, y)

# Train the model using the hyperparameters chosen
#train(X_train_raw, y_train, X_test_raw, y_test, k, lamda)


# Train the simple regression model for different hyperparameters
#train_logistic_regressor(X_train_raw, y_train)


reg_coeff = 10
simple_regression = train_once(X_train_raw, y_train, reg_coeff)

training_loss = log_loss(y_train, simple_regression.predict_proba(X_train_raw))
testing_loss = log_loss(y_test, simple_regression.predict_proba(X_test_raw))

y_pred = simple_regression.predict(X_test_raw)
accuracy = accuracy_score(y_test, y_pred)


# For testing
print(f"Training loss: {training_loss}")
print(f"Testing loss: {testing_loss}")
print(f"accuracy: {accuracy}")

#print(X_train_compressed.shape)
#print(X_test_compressed.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(X_inverse.shape)

