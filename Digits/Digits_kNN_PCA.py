import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


def main():

    # load dat use pandas
    data = pd.read_csv('test_8x8data_1-5.csv')
    # use first 75% for training and remaining 25% for testing
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = data.shape[0]
    # calculate the number of rows for training
    train_set_size = int(num_rows * train_ratio)

    # split features and label columns, library method
    # X, y = data.drop('digit', axis=1, inplace=False), data.loc[:, 'digit']
    # further split into 0.75, 0.25 ratio
    # train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=.25, shuffle=False)
    # train_features, test_features = X.iloc[:train_set_size], X.iloc[train_set_size:]
    # train_labels, test_labels = y[:train_set_size], y[train_set_size:]

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

    # set pca with 2 components
    pca = PCA(n_components=2)
    # fit and transform train features with pca of 2 component
    train_features_new = pca.fit_transform(train_features)
    # fit test features with pca
    test_features_new = pca.transform(test_features)

    # k-NN classification
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features_new, train_labels)
    # accuracy and error rate of the model
    score = knn.score(test_features_new, test_labels)
    print('Error rate of 5-NN classification with PCA at n_component=2 is ', (1-score))

    # # Alternative way to manually calculate error rate via values comparison
    # y_pred = knn.predict(test_features_new)
    # test_label_array = test_labels.to_numpy()
    # print(y_pred[0], test_label_array[0])
    # result = [0 if y_pred[i] == test_label_array[i] else 1 for i in range(len(y_pred))]
    # print('error rate:', sum(result)/ len(result))

    # plot decision boundaries of the k-NN classifier
    plot_decision_regions(train_features_new, train_labels, knn)
    plt.xlabel('First PC')
    plt.ylabel('Second PC')
    plt.title('Decision region plot of k-NN classification with k = 5')
    plt.legend()
    plt.show()

    # prepare scree plot
    pca_n = PCA(n_components=None)
    pca_n.fit_transform(train_features)

    # get PVE from PCA attribute 'explained_variance_ratio_'
    y_list = pca_n.explained_variance_ratio_
    x_list = list(range(1, len(y_list)+1))
    plt.plot(x_list, y_list, 'bo--')
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # axis labels and title
    plt.xlabel('$n^{th}$ Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot of PVE of Principal Components in PCA')
    plt.show()

    # select top 11 features using PCA
    print('The "elbow" of the scree plot occurs at 11th principal component')
    pca_11 = PCA(n_components=11)
    train_features_11_new = pca_11.fit_transform(train_features)
    test_features_11_new = pca_11.transform(test_features)
    # k-NN classification
    knn_2 = KNeighborsClassifier(n_neighbors=5)
    knn_2.fit(train_features_11_new, train_labels)
    new_score = knn_2.score(test_features_11_new, test_labels)
    print('Error rate of 5-NN classification with PCA at n_component=11 is ', (1-new_score))


if __name__ == '__main__':
    main()