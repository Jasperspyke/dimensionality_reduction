import data_module
from data_module import data1, data2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC


def sort_by_values(data):
    """Sort the eigenvector in descending order, so that later we can take the first two and just use those. """
    # Sort the data by the first element of each tuple (values) in descending order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)

    return sorted_data

def get_covariance_matrix_eigenpairs(data):

    """Ax = λx"""
    data[-500:, :] = data[-500:, :] - np.mean(data[-500:, :], axis=0)
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    unique_eigenvalues, counts = np.unique(eigenvalues, return_counts=True)

    if any(count > 1 for count in counts):
        raise ValueError
    eigenpairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]

    return eigenpairs


def project_and_plot(data, vec1, vec2):

    """The chosen eigenvectors will become the new basis for a lower-dimensional data representation.
     We're also going to plot in matplotlib and add a little rizz with seaborn for good measure."""
    # Make sure vec1 and vec2 are numpy arrays
    vec1 = np.array(vec1)
    # re-add mean
    vec2 = np.array(vec2)

    # Project the high-dimensional dataset onto vec1 and vec2
    projected_data = np.column_stack((data @ vec1, data @ vec2))

    projected_data[-500:, :] = (projected_data[-500:, :] + data_module.mean_info)
    sns.set(style="white", color_codes=True)
    sns.set_palette("husl")
    plt.grid()
    plt.scatter(projected_data[:500, 0], projected_data[:500, 1], marker='o', label='Normal')
    plt.scatter(projected_data[500:, 0], projected_data[500:, 1], marker='s', label='Abnormal')
    plt.xlabel("Projection onto vec1")
    plt.ylabel("Projection onto vec2")
    plt.title("High-Dimensional Data Projected onto 2D Space")
    plt.legend()
    plt.show()

    pca_df = pd.DataFrame(projected_data, columns=['vec1', 'vec2'])
    pca_df['class'] = 'Normal'
    pca_df.loc[500:, 'class'] = 'Abnormal'

    pca_df = pd.DataFrame(projected_data, columns=['vec1', 'vec2'])
    pca_df['class'] = 'Normal'
    pca_df.loc[500:, 'class'] = 'Abnormal'
    return pca_df


def svm_rbf_plot(pca_df):
    # Prepare the feature matrix (X) and target vector (y)
    X = pca_df[['vec1', 'vec2']].values
    y = pca_df['class'].values

    # Convert the class labels to integers
    _, y_int = np.unique(y, return_inverse=True)

    # Train the SVM classifier with an RBF kernel
    svm_clf = SVC(kernel='rbf', gamma='auto')
    svm_clf.fit(X, y_int)

    # Set the Seaborn style
    sns.set(style="white", color_codes=True)
    sns.set_palette("husl")

    # Generate the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the data points
    sns.scatterplot(x='vec1', y='vec2', hue='class', style='class', data=pca_df, markers=["o", "s"], edgecolor='k', linewidth=0.5)

    plt.xlabel("Projection onto vec1")
    plt.ylabel("Projection onto vec2")
    plt.title("SVM with RBF Kernel")
    plt.legend()
    plt.show()



if __name__=='__main__':
    data = np.concatenate((data1,data2),axis=0)
    eigenpairs = get_covariance_matrix_eigenpairs(data)
    eigenpairs = sort_by_values((eigenpairs))
    print(eigenpairs)

    # select the first two eigenpairs as our new 2d basis!
    eig1, eig2 = eigenpairs[0][1], eigenpairs[1][1]

    pca_data = project_and_plot(data, eig1, eig2)
    print(pca_data)

    svm_rbf_plot(pca_data)

