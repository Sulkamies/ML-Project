from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import numpy as np
import tqdm as tqdm
from data_split import trainTestSplit, dataImport

def train_logistic_regressor(X_train_raw, y_train):

    #x_train_normalized, y = dataImport()

    #X_train_raw, X_test_raw, y_train, y_test = trainTestSplit(x_train_normalized, y)

    regularization_coeff = np.logspace(-3, 2, 6)

    validation_errors = []
    training_errors = []
    accuracies = []

    for reg in tqdm.tqdm(regularization_coeff, desc= "Validating hyperparameters"):

        model = LogisticRegression(C = 1/reg, max_iter=1000)

        # Run a 5-fold cross validation
        results = cross_validate(
            model, 
            X_train_raw, 
            y_train, 
            cv=5, 
            return_train_score=True, 
            scoring= ('accuracy', 'neg_log_loss'),
            verbose=3,
            n_jobs=-2
        )
    
        training_errors.append(-1*results['train_neg_log_loss'].mean())
        validation_errors.append(-1*results['test_neg_log_loss'].mean())
        accuracies.append(results['test_accuracy'].mean())

    print(regularization_coeff)
    print(training_errors)
    print(validation_errors)

def train_once(X_train_raw, y_train, reg_coeff):
    model = LogisticRegression(C = 1/reg_coeff, max_iter=1000)
    model.fit(X_train_raw, y_train)

    return model






