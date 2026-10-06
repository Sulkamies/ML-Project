from sklearn.neural_network import MLPClassifier
from pca import pca_train, pca_transformation


def train(X_train_raw, y_train, X_test_raw, y_test, k, lamda) :
    print("Training initialized")

    pca = pca_train(X_train_raw, k)
    X_train_compressed = pca_transformation(X_train_raw, pca)
    X_test_compressed = pca_transformation(X_test_raw, pca)

    model = MLPClassifier(hidden_layer_sizes = (60, ), 
                      activation = "logistic", 
                      alpha = lamda, 
                      solver = "sgd", 
                      batch_size = 128,
                      learning_rate = "adaptive",
                      max_iter = 300)

    model.fit(X_train_compressed, y_train, sample_weight = None)

    y_train_predicted = model.predict(X_train_compressed)
    y_test_predicted = model.predict(X_test_compressed)

    print(f"Training accuracy: {accuracy(y_train_predicted, y_train)}")
    print(f"Test accuracy: {accuracy(y_test_predicted, y_test)}")

def accuracy(y_predicted, y) : return sum((y_predicted == y).astype(int).tolist()) / len(y_predicted)



