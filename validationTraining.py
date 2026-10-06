from pca import pca_transformation, pca_train
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

def validationAccuracy(X, y, split, lamda) :
    validation_accuracy = []
    validation_errors = []
    training_errors = []

    for train_index, val_index in split :

        # Extract the balanced data for this specific fold
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = y[train_index], y[val_index]

        model = MLPClassifier(hidden_layer_sizes = (60, ), 
                              activation = "logistic", 
                              alpha = lamda, 
                              solver = "sgd", 
                              batch_size = 128,
                              learning_rate = "adaptive",
                              max_iter = 100)

        model.fit(X_fold_train, y_fold_train, sample_weight = None)

        y_predicted = model.predict(X_fold_val)

        validation_accuracy.append(sum((y_predicted == y_fold_val).astype(int).tolist()) / len(y_predicted))
        validation_errors.append(log_loss(y_fold_val, model.predict_proba(X_fold_val)))
        training_errors.append(log_loss(y_fold_train, model.predict_proba(X_fold_train)))

    print(sum(validation_accuracy) / len(validation_accuracy))
    print(sum(validation_errors) / len(validation_errors))
    print(sum(training_errors) / len(training_errors))

    return sum(validation_accuracy) / len(validation_accuracy), sum(validation_errors) / len(validation_errors), sum(training_errors) / len(training_errors)


def gridSearch(X_train_raw, y, split) : 
    cells = []

    k_components = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    lamdas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

    for k in k_components :
        pca = pca_train(X_train_raw, k)
        X_train_compressed = pca_transformation(X_train_raw, pca)
        
        for lamda in lamdas :
            accuracy, val_loss, train_loss = validationAccuracy(X_train_compressed, y, split, lamda)
            cells.append((train_loss, val_loss, accuracy, k, lamda))

            print("Training completed")

    print(cells)
