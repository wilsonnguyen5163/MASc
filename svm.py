from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


class SVM_sgd:
    """
    SVM classification using Gradient Descent Method
    """

    def __init__(self, alpha=0.001, lmbda=0.01, passes=1000):
        self.alpha = alpha
        self.lmbda = lmbda
        self.passes = passes

    def fit(self, X_train: np.array, Y_train: np.array):
        """
        Tries to produce best hyperplane that separate 2 classes, by maximizing the margin 2/||w|| one sample at a time \n

        Margin = 2/||w|| \n
        Hyperplane = w * x + b = 0 \n

        ==> minimizes  lambda * ||w||^2 by iteratively update w to move along its direction of greatest descent
            Recall gradient of the cost function = 2 * lambda w
        """
        num_samples, num_features = X_train.shape

        # Length of <w> = length of training data feature's length
        self.w = np.zeros(num_features)
        self.b = 0

        # Make labels either 1, or -1 representing the 2 classes
        Y_train = np.where(Y_train <= 0, -1, 1)

        for _ in range(self.passes):
            for i in range(num_samples):
                # Check if current hyperplane satisfy the sample (correctly label the sample)
                margin_condition = Y_train[i] * \
                    (np.dot(X_train[i], self.w) + self.b) >= 1

                # If satisfied, no penalization, only regularize w to avoid overfitting
                if margin_condition:
                    self.w -= self.alpha * 2 * self.lmbda * self.w

                # Not satisfied, penalize the weight and bias term b based on the hinge loss dot product of (x_i and y_i)
                else:
                    self.w -= self.alpha * \
                        (2 * self.lmbda * self.w -
                         np.dot(X_train[i], Y_train[i]))
                    self.b -= self.alpha * Y_train[i]

    def predict(self, X_test):
        """
        Make classification prediction based on which side of the hyperplane the test point falls on 
        """
        return np.where(np.sign(np.dot(X_test, self.w) + self.b) < 0, 0, 1)


class SVM_qp:
    def __init__(self):
        pass


def svm_benchmark(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm_model.predict(X_test)
    return svm_model, y_pred


def svm_accuracy(y_test, y_prediction):
    accuracy = accuracy_score(y_test, y_prediction)
    print(f'Accuracy: {accuracy:.2f}')


def main():
    iris = datasets.load_iris()
    X, y = iris.data[iris.target != 2], iris.target[iris.target != 2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    benchmark_model, benchmark_res = svm_benchmark(X_train, X_test, y_train)

    my_model = SVM_sgd()
    my_model.fit(X_train, y_train)
    my_model_res = my_model.predict(X_test)
    print(benchmark_res)
    print(my_model_res)


if __name__ == "__main__":
    main()
