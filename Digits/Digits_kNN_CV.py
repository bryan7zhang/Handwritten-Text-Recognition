from sklearn.model_selection import KFold  # used for CV K-fold data set division only
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kNN(x, Xtrain, Ytrain, k):
    """
    k-Nearest Neighbours
    :param x: a test sample
    :param Xtrain: array of input vectors of the training set
    :param Ytrain: array of output values of the training set
    :param k: number of nearest neighbours
    :return: y_pred: predicted value for the test sample
    """

    # obtain euclidean distance list
    eu_distance_list = [np.linalg.norm(x - x_i) for x_i in Xtrain]
    eu_dict = {}
    y_votes = []
    # store index-value pairs in a dictionary for faster comparison and index tracking
    for i in range(len(eu_distance_list)):
        eu_dict[i] = eu_distance_list[i]
    # get list of k y values corresponding to smallest kth euclidean distance
    for j in range(k):
        min_index = min(eu_dict, key=eu_dict.get)
        y_votes.append(Ytrain[min_index])
        eu_dict.pop(min_index)
    # y_pred takes majority vote
    y_pred = max(set(y_votes), key=y_votes.count)
    return y_pred


# ample data
sample_x = np.array([[1, 1], [2, 3], [3, 2], [3, 4], [2, 5]])
sample_y = np.transpose(np.array([0, 0, 0, 1, 1]))


# sample_x = {'col1': [1, 2, 3, 3, 2], 'col2': [1, 3, 2, 4, 5]}
# df_sample_x = pd.DataFrame(data=sample_x)
# sample_y = {'y': [0, 0, 0, 1, 1]}
# df_sample_y = pd.DataFrame(data=sample_y)


def cv_knn(X, Y, k, k_fold):
    """

    :param X: attributes data frame
    :param Y: output data frame
    :param k: number of nearest neighbours
    :param k_fold: number of folds for CV
    :return:
    """
    kf = KFold(n_splits=k_fold)
    err_cv_list = []
    # prepare for K-fold CV while utilizing kNN function
    for train_index, validation_index in kf.split(list(range(len(Y)))):
        # divide training into training set and validation set
        train_x = X[train_index]
        train_y = Y[train_index]
        cv_x = X[validation_index]
        cv_y = Y[validation_index]
        # container for indicator function output of each CV trial
        i_list = []

        # compute misclassification rate for q-th trial
        for i in range(len(cv_x)):
            x_i = cv_x[i]
            y_i = cv_y[i]
            y_pred = kNN(x_i, train_x, train_y, k)
            i_list.append((0 if y_i == y_pred else 1))
        err_cv_list.append(sum(i_list) / len(i_list))
    return sum(err_cv_list) / len(err_cv_list)


def sk_cv_knn(X, Y, k, k_fold):
    """

    :param X: attributes data frame
    :param Y: output data frame
    :param k: number of nearest neighbours
    :param k_fold: number of folds for CV
    :return:
    """
    kf = KFold(n_splits=k_fold)
    err_cv_list = []
    # prepare for K-fold CV while utilizing kNN function
    for train_index, validation_index in kf.split(list(range(len(Y)))):
        # divide training into training set and validation set
        train_x = X[train_index]
        train_y = Y[train_index]
        cv_x = X[validation_index]
        cv_y = Y[validation_index]

        # deploys sklearn K neighbor classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # container for indicator function output of each CV trial
        i_list = []
        # compute misclassification rate for q-th trial
        for i in range(len(cv_x)):
            x_i = cv_x[i]
            y_i = cv_y[i]
            knn.fit(train_x, train_y)
            y_pred = KNeighborsClassifier.predict(knn, x_i.reshape(1, -1))
            i_list.append((0 if y_i == y_pred else 1))
        err_cv_list.append(sum(i_list) / len(i_list))
    return sum(err_cv_list) / len(err_cv_list)


def handwritten_classification():
    # Classification of Handwritten Digits
    # load data using pandas
    data = pd.read_csv("digits_8x8data_0-9.csv")
    # use first 75% for training and remaining 25% for testing
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = data.shape[0]
    # calculate the number of rows for training
    train_set_size = int(num_rows * train_ratio)
    # divide training set and test set
    train_data = data.iloc[:train_set_size]
    test_data = data.iloc[train_set_size:]
    print(len(train_data), "training samples + ", len(test_data), "test samples")

    # prepare training features and training labels
    train_features = train_data.drop('digit', axis=1, inplace=False)
    train_labels = train_data.loc[:, 'digit']

    # prepare test features and test labels
    test_features = test_data.drop('digit', axis=1, inplace=False)
    test_labels = test_data.loc[:, 'digit']

    k_fold = 10
    k_list = [(2 * n + 1) for n in range(11)]
    print(f"k = {k_list}")
    # compute cv estimates for each k
    cv_err = [sk_cv_knn(X=train_features.values, Y=train_labels.values, k=i, k_fold=k_fold)
              for i in k_list]
    # plot CV Estimates vs. k values
    line_plot(x_coord=k_list, cv_error=cv_err,
              title="CV Estimates of Classification of Handwritten Digits\nby k-NN",
              x_lb='k', y_lb='Prediction Error', x_l=1, x_h=23)
    # cv_err_formatted_list = ['%.4f' % elem for elem in cv_err]
    print(f"Corresponding CV Estimates: {cv_err}")

    cv_min_index = [i for i, x in enumerate(cv_err) if x == min(cv_err)]
    best_k = [k_list[i] for i in cv_min_index]
    print(f"Best: k = {best_k} produces the lowest prediction error")

    # compute
    for k in best_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        test_pred = KNeighborsClassifier.predict(knn, test_features)
        err_list = []
        for i in range(len(test_pred)):
            err_list.append((0 if test_pred[i] == test_labels.values[i] else 1))
        err_rate = sum(err_list) / len(err_list)
        print(f"The error rate of {k}-NN classifier on MNIST dataset is {err_rate}")


def line_plot(x_coord, cv_error, title='', x_lb='', y_lb='', x_l=None, x_h=None):
    """
    generates line plots
    :param x_coord: x coordinates of data points on the curve to be plotted
    :param cv_error: cv error as y coordinates
    :param title: title of the plot
    :param x_lb: label of x axis
    :param y_lb: label of y axis
    :param x_l: lower limit of y axis range
    :param x_h: upper limit of y axis range
    """
    plt.plot(x_coord, cv_error, 'red', label="CV Error")
    plt.legend()
    plt.title(title)
    plt.xlabel(x_lb)
    plt.ylabel(y_lb)
    # plt.xlim(x_l, x_h)
    plt.xticks(np.arange(x_l, x_h, 2))
    plt.show()


def main():
    # PART 2
    print("------------- PART 2 -------------")
    knn_result = cv_knn(sample_x, sample_y, k=3, k_fold=5)
    sk_result = sk_cv_knn(sample_x, sample_y, k=3, k_fold=5)
    print("The average LOOCV estimate of prediction error of sample data, implementing k-NN with\n"
          f"k = 3 is {knn_result}, which agrees with the calculation in part 1, 0.6.")
    print("The average LOOCV estimate of prediction error of sample data, implementing sklearn\n"
          f"KNeighbor classifier with n_neighbor=3 is {sk_result}, which agrees with kNN() implementation.")

    # PART 3
    print("\n------------- PART 3 -------------")
    handwritten_classification()


if __name__ == '__main__':
    main()
